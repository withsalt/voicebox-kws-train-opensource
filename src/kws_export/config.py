"""导出配置：读取 config.toml 的 [export] 段（缺省也能跑，带合理默认值）。

第八步把第七步训练出的「最优解」checkpoint 导出为**带显式 cache 的流式 ONNX 模型** + 配套
部署资源（tokens/lexicon/am.mvn/meta），供 C# 等运行时低延迟流式唤醒。

结构维度（层数/proj/lorder/rorder/输入输出维）不在此配置——导出时从基线模型
configuration.json 的 model.backbone 读出，跟随基线模型，避免与配置漂移。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExportConfig:
    enabled: bool = True
    opset: int = 17                # torch.onnx.export 的 opset
    work_subdir: str = "training"  # 训练产物目录（与 [train].work_subdir 对齐，用于找最优 checkpoint）
    num_avg: int = 10              # 首选导出的最优平均模型 avg_<num_avg>.pt（与 [train].num_avg 对齐）
    out_subdir: str = "onnx"       # 导出落 output/<词>/<out_subdir>/
    onnx_name: str = "kws_streaming.onnx"
    verify: bool = True            # 导出后跑 onnxruntime 不规则分块 vs torch 流式 parity
    self_check_tol: float = 1e-4   # 自检：逐帧流式 vs 整段前向 的最大允许误差
    ort_tol: float = 1e-4          # onnxruntime vs torch 流式 的最大允许误差


def load_export_config(path: str = "config.toml") -> ExportConfig:
    cfg = ExportConfig()
    p = Path(path)
    if p.exists():
        with p.open("rb") as f:
            data = tomllib.load(f)
        e = data.get("export", {})
        cfg.enabled = e.get("enabled", cfg.enabled)
        cfg.opset = int(e.get("opset", cfg.opset))
        cfg.work_subdir = e.get("work_subdir", cfg.work_subdir)
        cfg.num_avg = int(e.get("num_avg", cfg.num_avg))
        cfg.out_subdir = e.get("out_subdir", cfg.out_subdir)
        cfg.onnx_name = e.get("onnx_name", cfg.onnx_name)
        cfg.verify = e.get("verify", cfg.verify)
        cfg.self_check_tol = float(e.get("self_check_tol", cfg.self_check_tol))
        cfg.ort_tol = float(e.get("ort_tol", cfg.ort_tol))

        # work_subdir/num_avg 默认跟随 [train]，除非 [export] 显式覆盖（保持「最优解」与训练产物一致）。
        t = data.get("train", {})
        if "work_subdir" not in e and "work_subdir" in t:
            cfg.work_subdir = t["work_subdir"]
        if "num_avg" not in e and "num_avg" in t:
            cfg.num_avg = int(t["num_avg"])
    return cfg
