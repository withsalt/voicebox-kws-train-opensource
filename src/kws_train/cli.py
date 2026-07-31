"""命令行：训练（在本地基线模型上 finetune + 评测）。需 GPU。

  # 训练（数据须事先备好：train_wav.scp/cv_wav.scp/test_wav.scp/merge_trans.txt，见 README）
  # 默认读 [train]，lr 已下调、带 cv_loss 早停
  python -m kws_train train 小木小木

  # 覆盖部分参数（轮数/学习率/早停耐心/不评测）
  python -m kws_train train 小木小木 --epochs 20 --lr 5e-5 --patience 5
  python -m kws_train train 小木小木 --no-eval

  # 仅查看：配置 / 数据是否就绪 / GPU 可用性
  python -m kws_train info 小木小木
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kws_common.paths import safe_filename

from .config import load_train_config


def _force_utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _dataset_dir(output_dir: str, wake: str) -> Path:
    return Path(output_dir) / safe_filename(wake)


def _resolve_gpu(arg) -> int:
    """--gpu 未给时自动探测：有 CUDA 用 0，否则 -1（CPU）。"""
    if arg is not None:
        return arg
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_train_config(args.config)
    if not cfg.enabled:
        print("训练步骤被 [train].enabled=false 关闭。", file=sys.stderr)
        return 1
    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.patience is not None:
        cfg.early_stop_patience = args.patience
    if args.num_avg is not None:
        cfg.num_avg = args.num_avg
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.workers is not None:
        cfg.workers = args.workers
    if args.target_fa_per_hour is not None:
        cfg.target_fa_per_hour = args.target_fa_per_hour
    if args.no_eval:
        cfg.eval_after_train = False

    dataset_dir = _dataset_dir(args.output_dir, args.wake)
    if not (dataset_dir / "train_wav.scp").exists():
        print(f"错误：{dataset_dir / 'train_wav.scp'} 不存在，"
              f"请先准备好 train_wav.scp 等 4 个数据文件（见 README 数据格式）。", file=sys.stderr)
        return 1

    gpu = _resolve_gpu(args.gpu)
    from .trainer import run_training, summarize
    manifest = run_training(args.wake, dataset_dir, cfg, gpu=gpu, config_path=args.config)
    print(summarize(manifest))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    cfg = load_train_config(args.config)
    print("== 训练配置 [train] ==")
    print(f"  enabled={cfg.enabled}  model_dir={cfg.model_dir}")
    print(f"  lr={cfg.lr:g}  max_epochs={cfg.max_epochs}  batch_size={cfg.batch_size}  "
          f"workers={cfg.workers}")
    print(f"  num_avg={cfg.num_avg}  early_stop_patience={cfg.early_stop_patience}  "
          f"min_delta={cfg.early_stop_min_delta:g}")
    print(f"  eval_after_train={cfg.eval_after_train}  work_subdir={cfg.work_subdir}")
    print(f"  target_fa_per_hour={cfg.target_fa_per_hour:g}  target_min_recall={cfg.target_min_recall:g}")
    if args.wake:
        d = _dataset_dir(args.output_dir, args.wake)
        print(f"== 数据集就绪检查（{args.wake}）==")
        for f in ("train_wav.scp", "cv_wav.scp", "test_wav.scp", "merge_trans.txt"):
            p = d / f
            print(f"  {'✓' if p.exists() else '✗'} {p}")
    print("== 设备 ==")
    try:
        import torch
        avail = torch.cuda.is_available()
        print(f"  CUDA 可用={avail}  设备数={torch.cuda.device_count()}")
        if avail:
            print(f"  GPU0={torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"  (torch 不可用: {e})")
    return 0


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    parser = argparse.ArgumentParser(prog="kws_train", description="唤醒词训练（finetune + 评测）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="在本地基线模型上 finetune 并评测")
    pt.add_argument("wake", help="唤醒词（如 小木小木）")
    pt.add_argument("--config", default="config.toml")
    pt.add_argument("--output-dir", default="./output", help="输出根（默认 ./output）")
    pt.add_argument("--epochs", type=int, default=None, help="训练轮数上限（覆盖 [train].max_epochs）")
    pt.add_argument("--lr", type=float, default=None, help="学习率（覆盖 [train].lr）")
    pt.add_argument("--patience", type=int, default=None, help="早停耐心（覆盖 [train].early_stop_patience）")
    pt.add_argument("--num-avg", type=int, default=None, help="评测前平均的最优 checkpoint 数")
    pt.add_argument("--batch-size", type=int, default=None)
    pt.add_argument("--workers", type=int, default=None)
    pt.add_argument("--gpu", type=int, default=None, help=">=0 用对应 GPU，-1 用 CPU；默认自动探测")
    pt.add_argument("--target-fa-per-hour", type=float, default=None,
                    help="自动选阈值的误唤醒/小时预算（覆盖 [train].target_fa_per_hour）")
    pt.add_argument("--no-eval", action="store_true", help="只训练不评测")
    pt.set_defaults(func=cmd_train)

    pi = sub.add_parser("info", help="查看训练配置/数据就绪/GPU")
    pi.add_argument("wake", nargs="?", default=None)
    pi.add_argument("--config", default="config.toml")
    pi.add_argument("--output-dir", default="./output")
    pi.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
