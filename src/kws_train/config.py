"""训练配置：读取 config.toml 的 [train] 段（缺省也能跑，带合理默认值）。

用 modelscope 的 KWS nearfield 训练器在本地基线模型(speech_charctc_kws_phone)上 finetune，
吃预先备好的 train_wav.scp / cv_wav.scp / test_wav.scp / merge_trans.txt（数据格式见 README）。

为缓解小数据过拟合，这里把 lr 默认下调到 1e-4（基线 configuration.json 是 1e-3），并加
cv_loss 早停（连续 early_stop_patience 个 epoch 未改善即停）。需 GPU。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    enabled: bool = True
    model_dir: str = "speech_charctc_kws_phone"  # 本地基线模型目录（含 train/base.pt 等）
    lr: float = 1e-4               # finetune 学习率（下调防过拟合；基线是 1e-3）
    max_epochs: int = 30           # 训练轮数上限（早停可能提前停）
    batch_size: int = 64           # 训练/评测 batch（小数据用 64 比 256 多几步更新）
    workers: int = 2               # dataloader 工作进程数
    num_avg: int = 10              # 评测前按 cv_loss 平均的最优 checkpoint 数
    early_stop_patience: int = 6   # cv_loss 连续多少 epoch 未改善即早停（<=0 关闭）
    early_stop_min_delta: float = 0.0  # 视为“改善”的最小降幅
    eval_after_train: bool = True  # 训练后是否评测出 score.txt + DET
    # —— 在线 SpecAugment（训练器唯一的在线增强，作用于 80 维 fbank）——
    # 基线 configuration.json 为长句大数据调的 max_t=50/max_f=30/num_f_mask=2，对「短叠词 +
    # 声调近音区分」过强：频率最多遮 ~75%（抹掉声调谐波/共振峰）、时间最多遮 ~1s（盖掉关键词）。
    # 下调更稳；想恢复基线行为把四个值设回 2/2/50/30 即可。这些值会写进 work_dir/config.json。
    spec_aug_num_t_mask: int = 2   # 时间掩码条数
    spec_aug_num_f_mask: int = 1   # 频率掩码条数（基线 2 → 1）
    spec_aug_max_t: int = 20       # 单条时间掩码最大帧数 ×10ms（基线 50 → 20，约半个音节）
    spec_aug_max_f: int = 10       # 单条频率掩码最大 mel 频带数（基线 30 → 10，保住声调判别）
    # —— cv 选择指标（早停 / best_epoch / checkpoint 平均共用一个标量）——
    # 基线只看总 cv_loss，而 cv 负:正≈5:1 → 总 loss 被负样本主导，早停/选模偏「拟合负样本」、
    # 易欠训召回。balanced=按正/负各自平均再加权（消除数量失衡）；pos=只看正样本（最偏召回）；
    # total=基线行为。缺正或缺负样本时安全回退 total。
    cv_select_metric: str = "balanced"   # "balanced" | "pos" | "total"
    cv_pos_weight: float = 0.5            # balanced 下正样本权重，负样本权重=1-该值
    work_subdir: str = "training"  # 工作目录（落在 output/<词>/<work_subdir>）
    # 评测后自动选阈值（从 DET stats 取工作点）回写部署 keywords.json：
    target_fa_per_hour: float = 0.5    # 误唤醒/小时预算：取满足该预算的最高召回阈值（达不到则回退误唤醒最低点）
    target_min_recall: float = 0.0     # 召回下限（仅提示；选中点低于它会告警，0=不提示）


def load_train_config(path: str = "config.toml") -> TrainConfig:
    cfg = TrainConfig()
    p = Path(path)
    if p.exists():
        with p.open("rb") as f:
            data = tomllib.load(f)
        t = data.get("train", {})
        cfg.enabled = t.get("enabled", cfg.enabled)
        cfg.model_dir = t.get("model_dir", cfg.model_dir)
        cfg.lr = float(t.get("lr", cfg.lr))
        cfg.max_epochs = int(t.get("max_epochs", cfg.max_epochs))
        cfg.batch_size = int(t.get("batch_size", cfg.batch_size))
        cfg.workers = int(t.get("workers", cfg.workers))
        cfg.num_avg = int(t.get("num_avg", cfg.num_avg))
        cfg.early_stop_patience = int(t.get("early_stop_patience", cfg.early_stop_patience))
        cfg.early_stop_min_delta = float(t.get("early_stop_min_delta", cfg.early_stop_min_delta))
        cfg.eval_after_train = t.get("eval_after_train", cfg.eval_after_train)
        cfg.work_subdir = t.get("work_subdir", cfg.work_subdir)
        cfg.spec_aug_num_t_mask = int(t.get("spec_aug_num_t_mask", cfg.spec_aug_num_t_mask))
        cfg.spec_aug_num_f_mask = int(t.get("spec_aug_num_f_mask", cfg.spec_aug_num_f_mask))
        cfg.spec_aug_max_t = int(t.get("spec_aug_max_t", cfg.spec_aug_max_t))
        cfg.spec_aug_max_f = int(t.get("spec_aug_max_f", cfg.spec_aug_max_f))
        cfg.cv_select_metric = str(t.get("cv_select_metric", cfg.cv_select_metric))
        cfg.cv_pos_weight = float(t.get("cv_pos_weight", cfg.cv_pos_weight))
        cfg.target_fa_per_hour = float(t.get("target_fa_per_hour", cfg.target_fa_per_hour))
        cfg.target_min_recall = float(t.get("target_min_recall", cfg.target_min_recall))
    return cfg
