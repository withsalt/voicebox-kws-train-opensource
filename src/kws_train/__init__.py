"""kws_train：训练。

在本地基线模型 speech_charctc_kws_phone 上用 modelscope KWS nearfield 训练器 finetune，吃
预先备好的 train/cv/test_wav.scp + merge_trans.txt（数据格式见 README），评测出 score.txt + DET。

为缓解小数据过拟合：学习率下调（[train].lr 默认 1e-4）+ cv_loss 早停。需 GPU。
纯逻辑（should_early_stop/parse_score_file/summarize）不依赖 modelscope；run_training 内部延迟导入。
"""

from __future__ import annotations

from .config import TrainConfig, load_train_config
from .threshold import (build_keywords_json, parse_stats_file,
                        select_operating_point, write_keywords_json)
from .trainer import (parse_score_file, run_training, should_early_stop,
                      summarize)

__all__ = [
    "TrainConfig", "load_train_config", "run_training", "summarize",
    "should_early_stop", "parse_score_file",
    "parse_stats_file", "select_operating_point", "build_keywords_json",
    "write_keywords_json",
]
