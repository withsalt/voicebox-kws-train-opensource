"""第七步训练：在本地基线模型上 finetune（modelscope KWS nearfield 训练器）+ 评测。

为缓解小数据过拟合，做了两件事：
  1) 学习率下调（由 [train].lr 写入 configs.train.optimizer.lr，默认 1e-4）；
  2) cv_loss 早停：`EarlyStopKWSNearfieldTrainer` 覆盖 modelscope 1.37.1 的 `train()` epoch
     循环，连续 early_stop_patience 个 epoch 未改善即提前停（其余逻辑——每 epoch 存
     <e>.pt/<e>.yaml、ReduceLROnPlateau——与原版一致）。

模块顶层只用标准库，纯函数（should_early_stop / parse_score_file / summarize）不依赖
modelscope，便于单测；所有 modelscope/torch 重依赖在 run_training / 工厂里**延迟导入**。
"""

from __future__ import annotations

import copy
import datetime
import os
import re
from pathlib import Path


# --------------------------- 纯逻辑（可单测，无重依赖） ---------------------------

def should_early_stop(cv_losses, patience, min_delta=0.0):
    """根据逐 epoch 的 cv_loss 序列判断是否应早停。

    Args:
        cv_losses: List[float]，按 epoch 顺序的 cv_loss（每个 epoch 一个）。
        patience: 连续多少个 epoch 未改善即早停；<=0 表示关闭早停。
        min_delta: 视为“改善”的最小降幅（cv_loss 需 < 当前最优 - min_delta）。

    Returns:
        (stop: bool, best_epoch: int)。best_epoch 为最优 cv_loss 所在 epoch 索引（0 基）。
        cv_losses 为空时返回 (False, -1)。
    """
    if not cv_losses:
        return False, -1
    best = cv_losses[0]
    best_idx = 0
    since = 0  # 距上次改善的 epoch 数
    for i in range(1, len(cv_losses)):
        if cv_losses[i] < best - min_delta:
            best = cv_losses[i]
            best_idx = i
            since = 0
        else:
            since += 1
    stop = patience > 0 and since >= patience
    return stop, best_idx


def _read_kv(path):
    """读 kaldi 风格两列文件（utt_id \t 内容）为 dict。"""
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t") if "\t" in line else line.split(None, 1)
            d[parts[0]] = parts[1] if len(parts) > 1 else ""
    return d


def parse_score_file(score_path, trans_path, keyword):
    """解析评测 score.txt（每行：utt_id detected/rejected ...），用转写区分正/负。

    正样本转写==关键词逐字（如「小 木 小 木」）；其余为负样本（无转写者本就不在
    score.txt 里，不计入）。返回召回与误唤醒计数。
    """
    kw = " ".join(keyword)
    trans = _read_kv(trans_path)
    pos_total = pos_hit = neg_total = neg_fa = 0
    with open(score_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            uid = parts[0]
            detected = len(parts) >= 2 and parts[1] == "detected"
            t = (trans.get(uid, "") or "").strip()
            if t == kw:
                pos_total += 1
                pos_hit += 1 if detected else 0
            else:
                neg_total += 1
                neg_fa += 1 if detected else 0
    recall = (pos_hit / pos_total) if pos_total else 0.0
    return {
        "pos_total": pos_total, "pos_hit": pos_hit, "recall": recall,
        "neg_total": neg_total, "neg_fa": neg_fa,
    }


def summarize(m: dict) -> str:
    """训练 + 评测结果的人读摘要。"""
    bcl = m.get("best_cv_loss")
    bcl_s = f"{bcl:.4f}" if isinstance(bcl, (int, float)) else str(bcl)
    lines = [
        f"训练汇总（唤醒词={m['keyword']}）：",
        f"  轮数 {m['epochs_run']}/{m['max_epochs']}  "
        f"早停={'是' if m.get('early_stopped') else '否'}  "
        f"best_epoch={m.get('best_epoch')}  best_cv_loss={bcl_s}  lr={m['lr']:g}",
    ]
    sa = m.get("spec_aug")
    if sa:
        lines.append(
            f"  cv选择={m.get('cv_select_metric', 'total')}  "
            f"SpecAug(t={sa['num_t_mask']}×{sa['max_t']}, f={sa['num_f_mask']}×{sa['max_f']})")
    ev = m.get("eval")
    if ev:
        lines.append(
            f"  评测：召回 {ev['pos_hit']}/{ev['pos_total']}"
            f"（{ev['recall'] * 100:.1f}%）  误唤醒 {ev['neg_fa']}/{ev['neg_total']}")
        lines.append(f"  score: {m.get('score_file')}")
        th = m.get("threshold")
        if th:
            flag = "" if th.get("met_target") else "（未达误唤醒预算，已回退误唤醒最低点）"
            lines.append(
                f"  工作点：threshold1={th['threshold']:.3f}{flag}  "
                f"召回 {th['recall']*100:.1f}%  误唤醒 {th['fa_per_hour']:.3f}/h  "
                f"(预算 ≤{th.get('target_fa_per_hour')}/h)")
            lines.append(f"  keywords.json: {m.get('keywords_json')}")
    else:
        lines.append("  （未评测）")
    if m.get("avg_checkpoint"):
        lines.append(f"  平均模型: {m['avg_checkpoint']}")
    return "\n".join(lines)


# --------------------------- 训练器（延迟导入 modelscope） ---------------------------

def _make_trainer_class():
    """构造带早停的 KWS 训练器子类（modelscope 在此延迟导入）。"""
    from modelscope.trainers.audio.kws_nearfield_trainer import KWSNearfieldTrainer

    class EarlyStopKWSNearfieldTrainer(KWSNearfieldTrainer):
        """KWSNearfieldTrainer + cv_loss 早停。train() 改编自 modelscope 1.37.1 同名方法，
        仅在 epoch 循环里加入早停判断（连续 patience 个 epoch cv_loss 未改善即停）。"""

        def __init__(self, *args, early_stop_patience=6, early_stop_min_delta=0.0,
                     keyword=None, cv_select_metric="total", cv_pos_weight=0.5, **kwargs):
            self.early_stop_patience = early_stop_patience
            self.early_stop_min_delta = early_stop_min_delta
            # cv 选择指标（见 config.TrainConfig.cv_select_metric）：按目标 token 串是否等于关键词
            # 把 cv 分成正/负两类分别求平均，消除 cv 负:正≈5:1 对总 cv_loss 的主导。
            self.keyword = keyword
            self.cv_select_metric = cv_select_metric
            self.cv_pos_weight = cv_pos_weight
            self.kw_token_ids = None  # 关键词 token id 元组，train() 内据 token_table 解析
            # 训练后供编排读取的统计
            self.epochs_run = 0
            self.best_epoch = -1
            self.best_cv_loss = float("nan")
            self.early_stopped = False
            super().__init__(*args, **kwargs)

        def _cv_eval(self, data_loader, device, kw_ids):
            """逐句 CTC loss 的交叉验证：按目标 token 串是否等于关键词，分正/负累计。

            返回 (total_avg, pos_avg, neg_avg, n_pos, n_neg)，均为每句平均 NLL，与基线
            executor_cv 的 cv_loss 同量纲（reduction='sum'/num_utts == 这里逐句 'none' 求均）。
            """
            import math
            import torch
            import torch.nn.functional as F

            self.model.eval()
            pos_sum = neg_sum = 0.0
            pos_n = neg_n = 0
            with torch.no_grad():
                for batch in data_loader:
                    _keys, feats, target, feats_lengths, target_lengths = batch
                    feats = feats.to(device)
                    target = target.to(device)
                    feats_lengths = feats_lengths.to(device)
                    target_lengths = target_lengths.to(device)
                    num_utts = feats_lengths.size(0)
                    if num_utts == 0:
                        continue
                    logits, _ = self.model(feats)
                    logits = logits.transpose(0, 1).log_softmax(2)  # (T, B, D)
                    losses = F.ctc_loss(
                        logits, target, feats_lengths, target_lengths,
                        reduction="none")  # (B,) 每句 NLL（未按长度归一，与基线一致）
                    tgt = target.cpu()
                    tlen = target_lengths.cpu()
                    for i in range(num_utts):
                        li = losses[i].item()
                        if not math.isfinite(li):
                            continue
                        seq = tuple(tgt[i][:tlen[i]].tolist())
                        if seq == kw_ids:
                            pos_sum += li
                            pos_n += 1
                        else:
                            neg_sum += li
                            neg_n += 1
            pos_avg = pos_sum / pos_n if pos_n else float("nan")
            neg_avg = neg_sum / neg_n if neg_n else float("nan")
            tot_n = pos_n + neg_n
            tot_avg = (pos_sum + neg_sum) / tot_n if tot_n else float("nan")
            return tot_avg, pos_avg, neg_avg, pos_n, neg_n

        def _select_cv(self, total, pos, neg, n_pos, n_neg):
            """据 cv_select_metric 合成早停/选模用标量；缺正或缺负时安全回退 total。"""
            import math
            m = self.cv_select_metric
            if m == "pos" and n_pos > 0 and math.isfinite(pos):
                return pos
            if (m == "balanced" and n_pos > 0 and n_neg > 0
                    and math.isfinite(pos) and math.isfinite(neg)):
                w = self.cv_pos_weight
                return w * pos + (1.0 - w) * neg
            return total

        def train(self, *args, **kwargs):
            import yaml
            import torch
            from torch import optim as optim
            from torch.utils.data import DataLoader
            from tensorboardX import SummaryWriter
            from modelscope.msdatasets.dataset_cls.custom_datasets.audio.kws_nearfield_dataset \
                import kws_nearfield_dataset
            from modelscope.utils.checkpoint import load_checkpoint, save_checkpoint
            from modelscope.utils.device import create_device
            from modelscope.trainers.audio.kws_utils.batch_utils import executor_cv, executor_train
            from modelscope.trainers.audio.kws_utils.model_utils import count_parameters
            from modelscope.trainers.audio.kws_utils.file_utils import query_token_set
            from modelscope.utils.logger import get_logger
            logger = get_logger()

            train_checkpoint = kwargs.get("checkpoint", None)
            if train_checkpoint is not None and os.path.exists(train_checkpoint):
                self.checkpoint = train_checkpoint
            else:
                self.checkpoint = os.path.join(self.model_dir, "train/base.pt")
            self.tensorboard_dir = kwargs.get("tensorboard_dir", "tensorboard")

            assert kwargs["train_data"], "please config train data in dict kwargs"
            assert kwargs["cv_data"], "please config cv data in dict kwargs"
            assert kwargs["trans_data"], "please config transcription data in dict kwargs"
            self.train_data = kwargs["train_data"]
            self.cv_data = kwargs["cv_data"]
            self.trans_data = kwargs["trans_data"]
            self.need_dump = kwargs.get("need_dump", False) and (self.rank == 0)

            train_conf = self.configs["preprocessor"]
            cv_conf = copy.deepcopy(train_conf)
            cv_conf["speed_perturb"] = False
            cv_conf["spec_aug"] = False
            cv_conf["shuffle"] = False

            dump_train_file = os.path.join(self.work_dir, "dump_train.txt")
            dump_cv_file = os.path.join(self.work_dir, "dump_cv.txt")
            self.train_dataset = kws_nearfield_dataset(
                self.train_data, self.trans_data, train_conf, self.token_table,
                self.lexicon_table, self.need_dump, dump_train_file, True)
            self.cv_dataset = kws_nearfield_dataset(
                self.cv_data, self.trans_data, cv_conf, self.token_table,
                self.lexicon_table, self.need_dump, dump_cv_file, True)

            self.train_dataloader = DataLoader(
                self.train_dataset, batch_size=None,
                pin_memory=kwargs.get("pin_memory", False),
                num_workers=self.configs.train.dataloader.workers_per_gpu,
                prefetch_factor=self.configs.train.dataloader.get("prefetch", 2))
            self.cv_dataloader = DataLoader(
                self.cv_dataset, batch_size=None,
                pin_memory=kwargs.get("pin_memory", False),
                num_workers=self.configs.evaluation.dataloader.workers_per_gpu,
                prefetch_factor=self.configs.evaluation.dataloader.get("prefetch", 2))

            self.model = self.build_model(self.configs)
            num_params = count_parameters(self.model)
            if self.rank == 0:
                logger.warning("the number of model params: {}".format(num_params))

            if self.checkpoint is not None and os.path.exists(self.checkpoint):
                load_checkpoint(self.checkpoint, self.model)
                info_path = re.sub(".pt$", ".yaml", self.checkpoint)
                infos = {}
                if os.path.exists(info_path):
                    with open(info_path, "r") as fin:
                        infos = yaml.load(fin, Loader=yaml.FullLoader)
            else:
                logger.warning("Training with random initialized params")
                infos = {}
            self.start_epoch = infos.get("epoch", -1) + 1
            self.configs["train"]["start_epoch"] = self.start_epoch

            lr_last_epoch = infos.get("lr", self.configs["train"]["optimizer"]["lr"])
            self.configs["train"]["optimizer"]["lr"] = lr_last_epoch

            device_name = kwargs.get("device", "gpu")
            if self.world_size > 1:
                device_name = f"cuda:{self.local_rank}"
            self.train_device = create_device(device_name)

            if self.world_size > 1:
                assert torch.cuda.is_available()
                self.model = self.model.to(self.train_device)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            else:
                self.model = self.model.to(self.train_device)

            if self.rank == 0:
                os.makedirs(self.work_dir, exist_ok=True)
                saved_config_path = os.path.join(self.work_dir, "config.yaml")
                with open(saved_config_path, "w") as fout:
                    fout.write(yaml.dump(self.configs.to_dict()))

            writer = None
            if self.rank == 0:
                os.makedirs(self.work_dir, exist_ok=True)
                writer = SummaryWriter(os.path.join(self.work_dir, self.tensorboard_dir))

            log_interval = self.configs["train"].get("log_interval", 10)
            optim_conf = self.configs["train"]["optimizer"]
            optimizer = optim.Adam(
                self.model.parameters(), lr=optim_conf["lr"],
                weight_decay=optim_conf["weight_decay"])
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6, threshold=0.01)

            if self.start_epoch == 0 and self.rank == 0:
                save_checkpoint(self.model, os.path.join(self.work_dir, "init.pt"),
                                None, None, None, False, True)

            training_config = {
                "grad_clip": optim_conf["grad_clip"],
                "grad_accum": optim_conf.get("grad_accum", 1),
                "log_interval": log_interval,
                "world_size": self.world_size,
                "rank": self.rank,
                "local_rank": self.local_rank,
            }

            # —— cv 选择指标：解析关键词 token id，用于把 cv 分正/负（见 _cv_eval/_select_cv）——
            use_split_cv = (self.cv_select_metric != "total" and bool(self.keyword))
            if use_split_cv:
                _, kw_idx = query_token_set(
                    self.keyword.replace(" ", ""), self.token_table, self.lexicon_table)
                self.kw_token_ids = tuple(kw_idx)
                if not self.kw_token_ids:
                    logger.warning("关键词 token 解析为空，cv 选择指标回退 total")
                    use_split_cv = False

            max_epoch = self.configs["train"]["max_epochs"]
            cv_losses = []
            final_epoch = None
            logger.info(
                "Start training (early stop patience=%d, min_delta=%g, cv_metric=%s, split_cv=%s)..."
                % (self.early_stop_patience, self.early_stop_min_delta,
                   self.cv_select_metric, use_split_cv))
            totaltime = datetime.datetime.now()
            for epoch in range(self.start_epoch, max_epoch):
                self.train_dataset.set_epoch(epoch)
                training_config["epoch"] = epoch

                lr = optimizer.param_groups[0]["lr"]
                logger.info("Epoch {} TRAIN info lr {}".format(epoch, lr))
                executor_train(self.model, optimizer, self.train_dataloader,
                               self.train_device, writer, training_config)
                # cv_loss 即用于早停/选模/平均的“选择标量”：split_cv 时按 cv_select_metric 从
                # 正/负分项合成（消除负样本主导）；否则回退基线 executor_cv（总 cv_loss）。
                cv_pos = cv_neg = float("nan")
                if use_split_cv:
                    cv_total, cv_pos, cv_neg, n_pos, n_neg = self._cv_eval(
                        self.cv_dataloader, self.train_device, self.kw_token_ids)
                    cv_loss = self._select_cv(cv_total, cv_pos, cv_neg, n_pos, n_neg)
                    logger.info(
                        "Epoch {} EVAL cv_select({})={:.6f}  [total {:.6f} | pos {:.6f}({}) "
                        "| neg {:.6f}({})]".format(
                            epoch, self.cv_select_metric, cv_loss, cv_total,
                            cv_pos, n_pos, cv_neg, n_neg))
                else:
                    cv_loss, cv_acc = executor_cv(self.model, self.cv_dataloader,
                                                  self.train_device, training_config)
                    logger.info("Epoch {} EVAL info cv_loss {:.6f}, cv_acc {:.2f}".format(
                        epoch, cv_loss, cv_acc))

                if self.rank == 0:
                    save_model_path = os.path.join(self.work_dir, "{}.pt".format(epoch))
                    save_checkpoint(self.model, save_model_path, None, None, None, False, True)
                    info_path = re.sub(".pt$", ".yaml", save_model_path)
                    # cv_loss 写选择标量，使 average_model(val_best) 与早停/best_epoch 选同一指标
                    with open(info_path, "w") as fout:
                        fout.write(yaml.dump(dict(epoch=epoch, lr=lr, cv_loss=cv_loss)))
                    if writer is not None:
                        writer.add_scalar("epoch/cv_loss", cv_loss, epoch)
                        writer.add_scalar("epoch/lr", lr, epoch)
                        if use_split_cv:
                            writer.add_scalar("epoch/cv_loss_pos", cv_pos, epoch)
                            writer.add_scalar("epoch/cv_loss_neg", cv_neg, epoch)

                final_epoch = epoch
                lr_scheduler.step(cv_loss)

                # —— 早停判断（新增）——
                cv_losses.append(float(cv_loss))
                stop, best_idx = should_early_stop(
                    cv_losses, self.early_stop_patience, self.early_stop_min_delta)
                self.best_epoch = self.start_epoch + best_idx
                self.best_cv_loss = cv_losses[best_idx]
                if stop:
                    logger.info(
                        "Early stop at epoch {} (best epoch {}, best cv_loss {:.6f}); "
                        "no improvement for {} epochs.".format(
                            epoch, self.best_epoch, self.best_cv_loss, self.early_stop_patience))
                    self.early_stopped = True
                    break

            if final_epoch is not None and self.rank == 0 and writer is not None:
                writer.close()
            self.epochs_run = (final_epoch - self.start_epoch + 1) if final_epoch is not None else 0

            totaltime = datetime.datetime.now() - totaltime
            logger.info("Total time spent: {:.2f} hours".format(totaltime.total_seconds() / 3600.0))

    return EarlyStopKWSNearfieldTrainer


def run_training(wake, dataset_dir, cfg, *, gpu=0, config_path="config.toml"):
    """编排：写 config.json → 建带早停的训练器 → 训练 →（可选）评测 → 解析结果。

    每次调用都从 base.pt、epoch 0 重新训练（先清空 work_dir 旧 checkpoint），不做断点续训。

    Args:
        wake: 唤醒词（如 小木小木）。
        dataset_dir: 数据集目录（含 train/cv/test_wav.scp 与 merge_trans.txt）。
        cfg: TrainConfig。
        gpu: >=0 用对应 GPU，-1 用 CPU。
        config_path: config.toml 路径（仅记录用）。
    Returns:
        manifest dict（含 epochs_run/best_epoch/early_stopped/eval/score_file/outputs）。
    """
    from modelscope.utils.config import Config

    dataset_dir = Path(dataset_dir)
    model_dir = os.path.abspath(cfg.model_dir)
    work_dir = dataset_dir / cfg.work_subdir
    work_dir.mkdir(parents=True, exist_ok=True)

    # 清理上一轮训练残留的 checkpoint/info，避免 average_model（按 work_dir 下 *.yaml 选优平均）
    # 把旧轮次的 <e>.pt/<e>.yaml 误纳入本轮平均（本轮总从 base.pt、epoch0 重新开始）。
    for pat in ("*.pt", "*.yaml"):
        for f in work_dir.glob(pat):
            try:
                f.unlink()
            except OSError:
                pass

    conf_json = os.path.join(model_dir, "configuration.json")
    if not os.path.exists(conf_json):
        raise FileNotFoundError(f"基线模型配置缺失：{conf_json}")
    base_cfg = Config.from_file(conf_json)
    base_cfg.train.max_epochs = cfg.max_epochs
    base_cfg.train.optimizer.lr = cfg.lr
    base_cfg.train.dataloader["workers_per_gpu"] = cfg.workers
    base_cfg.evaluation.dataloader["workers_per_gpu"] = cfg.workers
    base_cfg.preprocessor.batch_conf["batch_size"] = cfg.batch_size
    # —— 优化①：软化在线 SpecAugment（短叠词/声调近音场景，基线 50/30/2/2 过强）——
    base_cfg.preprocessor.spec_aug_conf["num_t_mask"] = cfg.spec_aug_num_t_mask
    base_cfg.preprocessor.spec_aug_conf["num_f_mask"] = cfg.spec_aug_num_f_mask
    base_cfg.preprocessor.spec_aug_conf["max_t"] = cfg.spec_aug_max_t
    base_cfg.preprocessor.spec_aug_conf["max_f"] = cfg.spec_aug_max_f
    config_file = os.path.join(work_dir, "config.json")
    base_cfg.dump(config_file)

    train_scp = str(dataset_dir / "train_wav.scp")
    cv_scp = str(dataset_dir / "cv_wav.scp")
    test_scp = str(dataset_dir / "test_wav.scp")
    trans = str(dataset_dir / "merge_trans.txt")
    for f in (train_scp, cv_scp, trans):
        if not os.path.exists(f):
            raise FileNotFoundError(f"训练所需文件缺失：{f}（请先跑第六步汇编）")

    Trainer = _make_trainer_class()
    trainer = Trainer(
        model=model_dir, work_dir=str(work_dir), cfg_file=config_file, seed=666,
        early_stop_patience=cfg.early_stop_patience,
        early_stop_min_delta=cfg.early_stop_min_delta,
        keyword=wake, cv_select_metric=cfg.cv_select_metric, cv_pos_weight=cfg.cv_pos_weight)

    device = "gpu" if gpu >= 0 else "cpu"
    trainer.train(train_data=train_scp, cv_data=cv_scp, trans_data=trans, checkpoint="",
                  tensorboard_dir="tensorboard", need_dump=True, device=device)

    manifest = {
        "keyword": wake, "work_dir": str(work_dir),
        "max_epochs": cfg.max_epochs, "lr": cfg.lr,
        "epochs_run": getattr(trainer, "epochs_run", cfg.max_epochs),
        "best_epoch": getattr(trainer, "best_epoch", -1),
        "best_cv_loss": getattr(trainer, "best_cv_loss", float("nan")),
        "early_stopped": getattr(trainer, "early_stopped", False),
        "cv_select_metric": cfg.cv_select_metric,
        "spec_aug": {
            "num_t_mask": cfg.spec_aug_num_t_mask, "num_f_mask": cfg.spec_aug_num_f_mask,
            "max_t": cfg.spec_aug_max_t, "max_f": cfg.spec_aug_max_f,
        },
        "avg_checkpoint": None, "score_file": None, "eval": None,
        "outputs": {"work_dir": str(work_dir)},
    }

    if cfg.eval_after_train:
        test_dir = work_dir / "test_dir"
        avg_pt = work_dir / f"avg_{cfg.num_avg}.pt"
        if not os.path.exists(test_scp):
            print(f"  ⚠ 跳过评测：缺少 {test_scp}（可能用了 --no-test）")
        elif manifest["epochs_run"] < 1:
            print("  ⚠ 跳过评测：没有可用的训练 checkpoint。")
        elif manifest["epochs_run"] >= 6:
            # 常规路径：用 modelscope evaluate 内置的最优 checkpoint 平均（min_epoch=5，跳过欠
            # 训练的早期 epoch——实测对 KWS 召回更友好，且会自动生成部署用 convert.kaldi.txt）。
            trainer.evaluate("", None, test_dir=str(test_dir), test_data=test_scp,
                             trans_data=trans, average_num=cfg.num_avg, gpu=gpu,
                             keywords=wake, batch_size=cfg.batch_size)
        else:
            # 早停过早（<6 epoch）：内置平均的 min_epoch=5 会无候选而报错，改用 min_epoch=0
            # 自己平均（候选不足 num_avg 时 average_model 退化为单个最优）后评测。
            from modelscope.trainers.audio.kws_utils.model_utils import average_model
            print(f"  训练仅 {manifest['epochs_run']} 轮（<6）：改用 min_epoch=0 选优平均。")
            average_model(dst_model=str(avg_pt), src_path=str(work_dir),
                          val_best=True, avg_num=cfg.num_avg, min_epoch=0)
            try:
                from modelscope.trainers.audio.kws_utils.model_utils import convert_to_kaldi
                convert_to_kaldi(trainer.build_model(trainer.configs), str(avg_pt), str(work_dir))
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠ convert_to_kaldi 跳过（仅影响部署文本，不影响评测）：{e}")
            trainer.evaluate(str(avg_pt), None, test_dir=str(test_dir), test_data=test_scp,
                             trans_data=trans, average_num=cfg.num_avg, gpu=gpu,
                             keywords=wake, batch_size=cfg.batch_size)

        if test_dir.exists():
            manifest["avg_checkpoint"] = str(avg_pt) if avg_pt.exists() else None
            score_file = test_dir / "score.txt"
            if score_file.exists():
                manifest["score_file"] = str(score_file)
                manifest["eval"] = parse_score_file(str(score_file), trans, wake)
                # —— 自动选阈值 + 回写部署 keywords.json（读 compute_det 产出的 stats 文件）——
                _select_threshold(wake, test_dir, model_dir, cfg, manifest)
            manifest["outputs"]["score_file"] = str(score_file)

    return manifest


def _select_threshold(wake, test_dir, model_dir, cfg, manifest):
    """评测后：从 DET stats 选工作点并写 keywords.json，结果记入 manifest（失败仅告警，不中断）。"""
    from glob import glob
    from .threshold import select_and_write

    stats = sorted(glob(os.path.join(str(test_dir), "stats*.txt")))
    if not stats:
        print("  ⚠ 未找到 DET stats 文件，跳过自动选阈值。")
        return
    template = os.path.join(model_dir, "runtime", "kwsbp_resource", "keywords.json")
    try:
        op = select_and_write(
            stats[0], keyword=wake, out_path=os.path.join(str(test_dir), "keywords.json"),
            target_fa_per_hour=cfg.target_fa_per_hour, min_recall=cfg.target_min_recall,
            template_path=template if os.path.exists(template) else None)
    except (ValueError, OSError) as e:
        print(f"  ⚠ 自动选阈值失败（不影响训练产物）：{e}")
        return
    manifest["threshold"] = op
    manifest["keywords_json"] = op["keywords_json"]
    flag = "" if op["met_target"] else "（未达预算，已回退）"
    print(f"  自动选阈值：threshold1={op['threshold']:.3f}{flag}  "
          f"召回 {op['recall']*100:.1f}%  误唤醒 {op['fa_per_hour']:.3f}/h")
    print(f"  部署 keywords.json：{op['keywords_json']}")
