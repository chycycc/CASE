from tqdm import tqdm
from copy import deepcopy
from tensorboardX import SummaryWriter
from torch.nn.init import xavier_uniform_

from src.utils.config import config
from src.utils.common import set_seed
from src.models.CASE.model import CASE
from src.utils.data.loader import prepare_data_seq
from src.models.common import evaluate, count_parameters, make_infinite

def make_model(vocab, emo_num, strategy_num):
    is_eval = config.test
    if config.model == "case":
        model = CASE(
            vocab,
            emotion_num=emo_num,
            strategy_num=strategy_num,
            is_eval=is_eval,
            model_file_path=config.model_file_path if is_eval else None,
        )
    model = model.to(config.device)
    
    # Intialization
    if not is_eval:
        for n, p in model.named_parameters():
            if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
                xavier_uniform_(p)

    print("# PARAMETERS", count_parameters(model))

    return model

def pretrain(model, train_set):
    pretrain_epoch = config.pretrain_epoch
    check_iter = 200
    try:
        model.train()
        writer = SummaryWriter(log_dir=config.save_path)
        weights_best = deepcopy(model.state_dict())
        data_iter = make_infinite(train_set)
        steps = len(train_set)
        for epoch in range(pretrain_epoch):
            for n_iter in tqdm(range(steps)):
                bow_loss = model.train_one_batch(next(data_iter), n_iter)
                writer.add_scalars("bow_loss", {"loss_train": bow_loss}, n_iter)
                if config.noam:
                    writer.add_scalars(
                        "lr", {"learning_rata": model.optimizer._rate}, n_iter
                    )
            weights_best = deepcopy(model.state_dict())
    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")
        model.save_model(0, 0)
        weights_best = deepcopy(model.state_dict())

    return weights_best


def train(model, train_set, dev_set):
    check_iter = 2000
    iters = 20000 if config.dataset=="ED" else 6000
    # check_iter = 1
    try:
        model.train()
        best_ppl = 1000
        best_score = 1000.0
        patient = 0
        # [V5 Trial 4 / Trial 4b] 自适应分类头冻结监控与最佳状态缓存 (BCF)
        best_freeze_metric = -1.0 if config.freeze_metric == "emo_acc" else 1e9
        freeze_patient = 0
        best_emo_head_state = deepcopy(model.emotion_linear.state_dict())
        best_freeze_step = 0
        entered_mature = False
        writer = SummaryWriter(log_dir=config.save_path)
        weights_best = deepcopy(model.state_dict())
        data_iter = make_infinite(train_set)
        for n_iter in tqdm(range(1000000)):
            bow_loss, kl_loss, mim_loss, ctx_loss, ppl, str_loss, str_acc, emo_loss, emo_acc, epcl_loss, dec_emo_loss = model.train_one_batch(next(data_iter), n_iter)
            writer.add_scalars("bow_loss", {"loss_train": bow_loss}, n_iter)
            writer.add_scalars("kl_loss", {"loss_train": kl_loss}, n_iter)
            writer.add_scalars("mim_loss", {"loss_train": mim_loss}, n_iter)
            writer.add_scalars("ctx_loss", {"loss_train": ctx_loss}, n_iter)
            writer.add_scalars("ppl", {"ppl_train": ppl}, n_iter)
            if config.dataset == "ESConv":
                writer.add_scalars("str_loss", {"loss_train": str_loss}, n_iter)
                writer.add_scalars("str_acc", {"str_acc_train": str_acc}, n_iter)
            else:
                writer.add_scalars("emo_loss", {"loss_train": emo_loss}, n_iter)
                writer.add_scalars("emo_acc", {"emo_acc_train": emo_acc}, n_iter)
                writer.add_scalars("epcl_loss", {"loss_train": epcl_loss}, n_iter)
                writer.add_scalars("dec_emo_loss", {"loss_train": dec_emo_loss}, n_iter)  # [V4 Trial 8] 记录 Decoder MIM 损失
            if config.noam:
                # [V4 Trial 8] 记录实际生效的 LR（含后半程线性衰减），而非 NoamOpt 中停滞的 _rate
                actual_lr = model.optimizer.optimizer.param_groups[0]["lr"]
                writer.add_scalars(
                    "lr", {"learning_rata": actual_lr}, n_iter
                )

            if (n_iter + 1) % check_iter == 0:
                model.eval()
                model.epoch = n_iter
                bow_loss_val, kl_loss_val, mim_loss_val, ctx_loss_val, ppl_val, str_loss_val, str_acc_val, emo_loss_val, emo_acc_val, _ = evaluate(
                    model, dev_set, ty="valid", max_dec_step=50
                )
                writer.add_scalars("bow_loss", {"bow_loss_valid": bow_loss_val}, n_iter)
                writer.add_scalars("kl_loss", {"kl_loss_valid": kl_loss_val}, n_iter)
                writer.add_scalars("mim_loss", {"mim_loss_valid": mim_loss_val}, n_iter)
                writer.add_scalars("ctx_loss", {"ctx_loss_valid": ctx_loss_val}, n_iter)
                writer.add_scalars("ppl", {"ppl_valid": ppl_val}, n_iter)
                if config.dataset == "ESConv":
                    writer.add_scalars("str_loss", {"str_loss_valid": str_loss_val}, n_iter)
                    writer.add_scalars("str_acc", {"str_acc_valid": str_acc_val}, n_iter)
                else:
                    writer.add_scalars("emo_loss", {"emo_loss_valid": emo_loss_val}, n_iter)
                    writer.add_scalars("emo_acc", {"emo_acc_valid": emo_acc_val}, n_iter)
                
                # [V5 Trial 4 / Trial 4b] 自适应分类头冻结逻辑监控与黄金回滚 (BCF)
                if config.adaptive_freeze and not model.is_frozen and config.dataset == "ED":
                    curr_step = n_iter + 1
                    cur_metric = emo_acc_val if config.freeze_metric == "emo_acc" else emo_loss_val
                    
                    if curr_step >= config.min_freeze_step:
                        is_improved = (cur_metric > best_freeze_metric) if config.freeze_metric == "emo_acc" else (cur_metric < best_freeze_metric)
                        if is_improved:
                            best_freeze_metric = cur_metric
                            best_emo_head_state = deepcopy(model.emotion_linear.state_dict())
                            best_freeze_step = curr_step
                            freeze_patient = 0
                            print(f"[Adaptive Freeze Monitor] Step {curr_step}: {config.freeze_metric} 创新高至 {best_freeze_metric:.4f}，缓存黄金分类头权重，重置耐心计数器。")
                        else:
                            freeze_patient += 1
                            print(f"[Adaptive Freeze Monitor] Step {curr_step}: {config.freeze_metric}={cur_metric:.4f} (历史最优={best_freeze_metric:.4f} at Step {best_freeze_step})，未创新高！Patience: {freeze_patient}/{config.freeze_patience}")
                            
                        if freeze_patient >= config.freeze_patience or curr_step >= config.max_freeze_step:
                            trigger_reason = f"验证集平台期触发 (Patience耗尽: {freeze_patient}/{config.freeze_patience})" if freeze_patient >= config.freeze_patience else f"达到最大步数兜底上限 ({curr_step}>={config.max_freeze_step})"
                            print(f"[Adaptive Freeze] 触发原因: {trigger_reason}")
                            model.freeze_emo_head(
                                curr_step,
                                best_state=best_emo_head_state if config.rollback_best_freeze else None,
                                best_step=best_freeze_step
                            )
                    else:
                        # 处于冷启动防抖期，更新当前最优值并缓存权重，不消耗耐心
                        is_cold_improved = (cur_metric > best_freeze_metric) if config.freeze_metric == "emo_acc" else (cur_metric < best_freeze_metric)
                        if is_cold_improved:
                            best_freeze_metric = cur_metric
                            best_emo_head_state = deepcopy(model.emotion_linear.state_dict())
                            best_freeze_step = curr_step
                        print(f"[Adaptive Freeze Monitor] Step {curr_step} < min_freeze_step({config.min_freeze_step})，处于冷启动防抖保护期，记录当前最优 {config.freeze_metric}={best_freeze_metric:.4f}")

                model.train()
                if n_iter < iters:
                    continue

                # [V8.1 路线 B / V8.2 D] 验证集保存判定（支持复合评分：acc 模式或帕累托 emo_loss 模式）
                if getattr(config, 'use_composite_score', False):
                    composite_mode = getattr(config, 'composite_mode', 'acc')
                    if composite_mode == 'emo_loss':
                        alpha = getattr(config, 'composite_emo_loss_weight', 8.0)
                        cur_score = ppl_val + alpha * emo_loss_val
                        score_info = f"Score={cur_score:.4f} (PPL={ppl_val:.4f}, EMO_loss={emo_loss_val:.4f})"
                    else:
                        composite_acc_weight = getattr(config, 'composite_acc_weight', 20.0)
                        cur_score = ppl_val - composite_acc_weight * emo_acc_val
                        score_info = f"Score={cur_score:.4f} (PPL={ppl_val:.4f}, EMO_acc={emo_acc_val:.4f})"
                    writer.add_scalars("composite_score", {"score_valid": cur_score}, n_iter)
                    is_score_better = (cur_score <= best_score)
                else:
                    cur_score = ppl_val
                    is_score_better = (ppl_val <= best_ppl)
                    score_info = f"PPL={ppl_val:.4f}"

                min_save_step = getattr(config, 'min_save_step', 0)
                in_protection_period = (min_save_step > 0 and n_iter < min_save_step)

                if in_protection_period:
                    # [V8.2 E] 处于退火成熟保护期：记录过渡指标，但不消耗早停耐心
                    patient = 0
                    if is_score_better:
                        best_score = cur_score
                        best_ppl = ppl_val
                        model.save_model(best_ppl, n_iter)
                        weights_best = deepcopy(model.state_dict())
                        print(f"[*] Step {n_iter} < min_save_step({min_save_step}): 保护期刷新过渡权重！{score_info}")
                    else:
                        print(f"[-] Step {n_iter} < min_save_step({min_save_step}): 保护期未改善 ({score_info})，早停耐心保持为 0")
                else:
                    # [V8.2 E] 进入退火成熟期：若此前处于保护期，首步重置成熟期黄金检查点基线
                    if min_save_step > 0 and not entered_mature:
                        entered_mature = True
                        best_score = cur_score
                        best_ppl = ppl_val
                        patient = 0
                        model.save_model(best_ppl, n_iter)
                        weights_best = deepcopy(model.state_dict())
                        print(f"[*] [V8.2 E] Step {n_iter}: 正式进入退火成熟期 (>= min_save_step {min_save_step})，锚定成熟期基线黄金检查点！{score_info}")
                    else:
                        if is_score_better:
                            best_score = cur_score
                            best_ppl = ppl_val
                            patient = 0
                            model.save_model(best_ppl, n_iter)
                            weights_best = deepcopy(model.state_dict())
                            print(f"[*] Step {n_iter}: 刷新成熟期黄金权重！{score_info}，已落盘检查点。")
                        else:
                            patient += 1
                            max_patience = getattr(config, "patience", 5)
                            print(f"[-] Step {n_iter}: 未改善 ({score_info} vs 最优={best_score:.4f})，Patience: {patient}/{max_patience}")
                        if patient >= getattr(config, "patience", 5):
                            print(f"[Early Stopping] 连续 {patient} 次评估未刷新最优指标，触发早停机制退出。")
                            break

    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")
        model.save_model(best_ppl, n_iter)
        weights_best = deepcopy(model.state_dict())

    return weights_best

def test(model, test_set):
    model.eval()
    model.is_eval = True
    bow_loss_test, kl_loss_test, mim_loss_test, ctx_loss_test, ppl_test, str_loss_test, str_acc_test, emo_loss_test, emo_acc_test, results = evaluate(
        model, test_set, ty="test", max_dec_step=50
    )
    file_summary = config.save_path + "/results.txt"
    with open(file_summary, "w", encoding="utf-8") as f:
        f.write("EVAL\tBOW_Loss\tKL_Loss\tMIM_Loss\tCTX_Loss\tPPL\tSTR_loss\tSTR_acc\tEMO_loss\tEMO_acc\n")
        f.write(
            "{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
                "test", bow_loss_test, kl_loss_test, mim_loss_test, ctx_loss_test, ppl_test, str_loss_test, str_acc_test, emo_loss_test, emo_acc_test
            )
        )
        for r in results:
            f.write(r)

def main():
    set_seed()  # for reproducibility

    train_set, dev_set, test_set, vocab, emo_num, strategy_num = prepare_data_seq(
        batch_size=config.batch_size
    )

    model = make_model(vocab, emo_num, strategy_num)

    if config.test:
        test(model, test_set)
    else:
        if config.pretrain:
            weights_best = pretrain(model, train_set)
            model.load_state_dict({name: weights_best[name] for name in weights_best})
            config.pretrain = False
        weights_best = train(model, train_set, dev_set)
        model.epoch = 1
        model.load_state_dict({name: weights_best[name] for name in weights_best})
        test(model, test_set)

if __name__ == '__main__':
    # prepare_data_seq(batch_size=config.batch_size)
    main()