"""kws_export：导出。

把 kws_train 训练出的「最优解」checkpoint（modelscope 格式）映射并导出为**带显式 cache 的
流式 ONNX 模型** + 部署资源（tokens/lexicon/am.mvn/kws_meta.json），供 C# 等运行时低延迟流式唤醒。

带精确 lookahead 的流式 wrapper。纯逻辑（remap_state_dict_keys/find_best_checkpoint/
keyword_token_ids/read_backbone_dims/build_meta/summarize）不依赖 funasr/onnx；
export_onnx 内部延迟导入重依赖。
"""

from __future__ import annotations

from .config import ExportConfig, load_export_config
from .exporter import (build_meta, export_onnx, find_best_checkpoint,
                       keyword_token_ids, read_backbone_dims,
                       remap_state_dict_keys, summarize)

__all__ = [
    "ExportConfig", "load_export_config", "export_onnx", "summarize",
    "remap_state_dict_keys", "find_best_checkpoint", "keyword_token_ids",
    "read_backbone_dims", "build_meta",
]
