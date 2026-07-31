"""命令行：导出（把训练最优 checkpoint 导出为流式 ONNX）。可单独执行。

  # 导出（训练产物由 kws_train 产出）；读 [export]/[train] 配置，自动挑最优 checkpoint
  python -m kws_export export 小木小木

  # 覆盖部分参数（opset / 不跑 onnxruntime 校验）
  python -m kws_export export 小木小木 --opset 17 --no-verify

  # 仅查看：配置 / 训练产物与最优 checkpoint 是否就绪 / onnx 依赖可用性
  python -m kws_export info 小木小木

需在装有 funasr + onnx + onnxruntime 的环境运行（见 requirements-train.txt）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kws_common.paths import safe_filename
from kws_train.config import load_train_config

from .config import load_export_config


def _force_utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _dataset_dir(output_dir: str, wake: str) -> Path:
    return Path(output_dir) / safe_filename(wake)


def cmd_export(args: argparse.Namespace) -> int:
    ecfg = load_export_config(args.config)
    if not ecfg.enabled:
        print("导出步骤被 [export].enabled=false 关闭。", file=sys.stderr)
        return 1
    if args.opset is not None:
        ecfg.opset = args.opset
    if args.no_verify:
        ecfg.verify = False

    tcfg = load_train_config(args.config)
    dataset_dir = _dataset_dir(args.output_dir, args.wake)
    training_dir = dataset_dir / ecfg.work_subdir
    if not training_dir.is_dir():
        print(f"错误：{training_dir} 不存在，请先用 kws_train 训练。", file=sys.stderr)
        return 1

    from .exporter import export_onnx, summarize
    print(f"导出流式 ONNX（唤醒词={args.wake}，opset={ecfg.opset}）：{dataset_dir / ecfg.out_subdir}")
    manifest = export_onnx(args.wake, dataset_dir, ecfg,
                           model_dir=tcfg.model_dir, config_path=args.config)
    print(summarize(manifest))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    ecfg = load_export_config(args.config)
    tcfg = load_train_config(args.config)
    print("== 导出配置 [export] ==")
    print(f"  enabled={ecfg.enabled}  opset={ecfg.opset}  verify={ecfg.verify}")
    print(f"  work_subdir={ecfg.work_subdir}  num_avg={ecfg.num_avg}  "
          f"out_subdir={ecfg.out_subdir}  onnx_name={ecfg.onnx_name}")
    print(f"  基线模型 model_dir（[train]）={tcfg.model_dir}")
    if args.wake:
        d = _dataset_dir(args.output_dir, args.wake)
        training_dir = d / ecfg.work_subdir
        print(f"== 训练产物就绪检查（{args.wake}）==")
        print(f"  {'✓' if training_dir.is_dir() else '✗'} 训练目录 {training_dir}")
        try:
            from .exporter import find_best_checkpoint
            p, kind, info = find_best_checkpoint(training_dir, ecfg.num_avg)
            print(f"  ✓ 最优 checkpoint：{kind}  {p}  {info}")
        except FileNotFoundError as e:
            print(f"  ✗ {e}")
    print("== 导出依赖 ==")
    for mod in ("torch", "funasr", "onnx", "onnxruntime"):
        try:
            m = __import__(mod)
            print(f"  ✓ {mod} {getattr(m, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {mod} 不可用：{e}")
    return 0


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    parser = argparse.ArgumentParser(prog="kws_export", description="唤醒词导出（训练最优解 → 流式 ONNX）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("export", help="把训练最优 checkpoint 导出为流式 ONNX + 部署资源")
    pe.add_argument("wake", help="唤醒词（如 小木小木）")
    pe.add_argument("--config", default="config.toml")
    pe.add_argument("--output-dir", default="./output", help="输出根（默认 ./output）")
    pe.add_argument("--opset", type=int, default=None, help="ONNX opset（覆盖 [export].opset）")
    pe.add_argument("--no-verify", action="store_true", help="跳过 onnxruntime parity 校验")
    pe.set_defaults(func=cmd_export)

    pi = sub.add_parser("info", help="查看导出配置 / 训练产物就绪 / 依赖可用性")
    pi.add_argument("wake", nargs="?", default=None)
    pi.add_argument("--config", default="config.toml")
    pi.add_argument("--output-dir", default="./output")
    pi.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
