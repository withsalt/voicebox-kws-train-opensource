"""kws_train 自测（纯逻辑，不跑训练、不依赖 modelscope/GPU）。

运行：python tests/test_kws_train.py
不依赖 pytest——纯 assert + 计数，结尾打印 PASS/FAIL 汇总。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from kws_train import (TrainConfig, build_keywords_json,  # noqa: E402
                       load_train_config, parse_score_file, parse_stats_file,
                       select_operating_point, should_early_stop, summarize)

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ✓ {name}")
    else:
        _failed += 1
        print(f"  ✗ {name}  {detail}")


def test_config():
    print("[CONFIG]")
    c = TrainConfig()
    check("默认 lr 已下调到 1e-4", abs(c.lr - 1e-4) < 1e-12, str(c.lr))
    check("默认早停耐心 6", c.early_stop_patience == 6)
    check("默认 eval_after_train", c.eval_after_train is True)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "config.toml"
        p.write_text(
            "[train]\nlr = 0.00005\nmax_epochs = 12\nearly_stop_patience = 3\n"
            "batch_size = 32\nenabled = false\n", encoding="utf-8")
        c2 = load_train_config(str(p))
        check("解析 lr", abs(c2.lr - 5e-5) < 1e-12, str(c2.lr))
        check("解析 max_epochs", c2.max_epochs == 12)
        check("解析 patience", c2.early_stop_patience == 3)
        check("解析 batch_size", c2.batch_size == 32)
        check("解析 enabled=false", c2.enabled is False)
    # 缺文件用默认
    c3 = load_train_config(str(Path(tempfile.gettempdir()) / "no_such_kwstrain.toml"))
    check("缺配置回退默认", c3.max_epochs == TrainConfig().max_epochs)


def test_early_stop():
    print("[EARLY-STOP]")
    # 空
    stop, best = should_early_stop([], 6, 0.0)
    check("空序列不停", stop is False and best == -1)
    # 单调下降，不应停，best=最后一个
    losses = [10, 9, 8, 7, 6, 5, 4]
    stop, best = should_early_stop(losses, 3, 0.0)
    check("单调下降不停", stop is False, str((stop, best)))
    check("单调下降 best=末位", best == len(losses) - 1, str(best))
    # 在 epoch2 触底后连续 3 个未改善 -> patience=3 应停
    losses = [10, 9, 8, 8.1, 8.2, 8.3]  # best idx=2，其后 3 个未改善
    stop, best = should_early_stop(losses, 3, 0.0)
    check("触底后耐心耗尽即停", stop is True, str((stop, best)))
    check("best=触底位 idx2", best == 2, str(best))
    # 同样序列但耐心=4 -> 还不停（只 3 个未改善）
    stop, best = should_early_stop(losses, 4, 0.0)
    check("耐心更大则不停", stop is False, str((stop, best)))
    # min_delta：微小下降不算改善
    losses = [10, 9.99, 9.98, 9.97]  # 每步降 0.01
    stop, best = should_early_stop(losses, 2, 0.05)  # 需降>0.05 才算改善
    check("min_delta 抑制微小改善->停", stop is True, str((stop, best)))
    check("min_delta 下 best=首位", best == 0, str(best))
    # patience<=0 关闭早停
    stop, _ = should_early_stop([10, 11, 12, 13, 14], 0, 0.0)
    check("patience<=0 永不停", stop is False)


def test_parse_score():
    print("[PARSE-SCORE]")
    with tempfile.TemporaryDirectory() as d:
        trans = Path(d) / "merge_trans.txt"
        score = Path(d) / "score.txt"
        # 3 正样本（小 木 小 木），2 命中 1 漏；2 负样本，1 误唤醒
        trans.write_text(
            "te_p0\t小 木 小 木\n"
            "te_p1\t小 木 小 木\n"
            "te_p2\t小 木 小 木\n"
            "te_n0\t今 天 天 气 不 错\n"
            "te_n1\t帮 我 导 航 回 家\n", encoding="utf-8")
        score.write_text(
            "te_p0 detected 小木小木 0.9\n"
            "te_p1 detected 小木小木 0.6\n"
            "te_p2 rejected\n"
            "te_n0 rejected\n"
            "te_n1 detected 小木小木 0.3\n", encoding="utf-8")
        r = parse_score_file(str(score), str(trans), "小木小木")
        check("正样本总数 3", r["pos_total"] == 3, str(r))
        check("命中 2", r["pos_hit"] == 2, str(r))
        check("召回 2/3", abs(r["recall"] - 2 / 3) < 1e-9, str(r))
        check("负样本总数 2", r["neg_total"] == 2, str(r))
        check("误唤醒 1", r["neg_fa"] == 1, str(r))


def test_summarize():
    print("[SUMMARIZE]")
    m = {
        "keyword": "小木小木", "max_epochs": 30, "lr": 1e-4,
        "epochs_run": 14, "best_epoch": 7, "best_cv_loss": 12.3456,
        "early_stopped": True,
        "eval": {"pos_hit": 8, "pos_total": 9, "recall": 8 / 9, "neg_fa": 0, "neg_total": 150},
        "score_file": "/x/score.txt", "avg_checkpoint": "/x/avg_10.pt",
    }
    s = summarize(m)
    check("含唤醒词", "小木小木" in s)
    check("含早停标记", "早停=是" in s, s)
    check("含召回", "8/9" in s, s)
    check("含 best_cv_loss", "12.3456" in s, s)
    # 未评测分支
    m2 = dict(m, eval=None, early_stopped=False)
    s2 = summarize(m2)
    check("未评测分支", "未评测" in s2 and "早停=否" in s2, s2)


def test_threshold():
    print("[THRESHOLD]")
    # 默认配置带选阈值字段
    c = TrainConfig()
    check("默认 target_fa_per_hour=0.5", abs(c.target_fa_per_hour - 0.5) < 1e-12, str(c.target_fa_per_hour))
    check("默认 target_min_recall=0", c.target_min_recall == 0.0)

    # 解析 stats 文件（threshold recall fa_rate fa_per_hour）
    with tempfile.TemporaryDirectory() as d:
        sp = Path(d) / "stats.小_木_小_木.txt"
        sp.write_text(
            "0.000 1.000000 0.500000 5.000000\n"
            "0.100 0.980000 0.200000 2.000000\n"
            "0.200 0.950000 0.040000 0.400000\n"   # 首个 fa<=0.5 的点（最高召回）
            "0.300 0.900000 0.010000 0.100000\n"
            "0.400 0.800000 0.000000 0.000000\n"
            "\n# 注释行/空行应被忽略\n", encoding="utf-8")
        rows = parse_stats_file(str(sp))
        check("解析行数=5", len(rows) == 5, str(len(rows)))
        check("升序首行 threshold=0", rows[0]["threshold"] == 0.0)

        # 预算 0.5/h：满足的有 thr>=0.2，取召回最高 = thr 0.2（召回0.95）
        op = select_operating_point(rows, target_fa_per_hour=0.5)
        check("达预算时选最高召回点 thr=0.2", abs(op["threshold"] - 0.2) < 1e-9, str(op))
        check("该点召回 0.95", abs(op["recall"] - 0.95) < 1e-9, str(op))
        check("met_target=True", op["met_target"] is True)
        check("误唤醒 0.4/h ≤ 预算", op["fa_per_hour"] <= 0.5)

        # 预算极严 0.0/h：仅 thr 0.4 满足（fa=0），选它
        op2 = select_operating_point(rows, target_fa_per_hour=0.0)
        check("极严预算选 fa=0 的 thr=0.4", abs(op2["threshold"] - 0.4) < 1e-9, str(op2))

        # 预算不可达（负数）：回退误唤醒最低点（thr 0.4, fa=0）
        op3 = select_operating_point(rows, target_fa_per_hour=-1.0)
        check("不可达预算回退 met_target=False", op3["met_target"] is False, str(op3))
        check("回退到误唤醒最低点 thr=0.4", abs(op3["threshold"] - 0.4) < 1e-9, str(op3))

        # min_recall 提示：选中点 0.95 < 0.99 -> low_recall=True
        op4 = select_operating_point(rows, target_fa_per_hour=0.5, min_recall=0.99)
        check("召回低于下限置 low_recall", op4["low_recall"] is True, str(op4))

    # 空 stats 抛错
    try:
        select_operating_point([], target_fa_per_hour=0.5)
        check("空 stats 应抛错", False)
    except ValueError:
        check("空 stats 应抛错", True)


def test_keywords_json():
    print("[KEYWORDS-JSON]")
    # 无模板：用内置兜底字段
    kj = build_keywords_json("小木小木", 0.2345)
    wl = kj["word_list"]
    check("word_list 单条", len(wl) == 1)
    check("name=唤醒词(不空格)", wl[0]["name"] == "小木小木", wl[0]["name"])
    check("threshold1 写入并保留6位", abs(wl[0]["threshold1"] - 0.2345) < 1e-9, str(wl[0]["threshold1"]))
    check("含部署字段 cc_name", "cc_name" in wl[0])
    check("type=wakeup", wl[0]["type"] == "wakeup")

    # 有模板：沿用模板其余字段，仅覆盖 name/threshold1
    with tempfile.TemporaryDirectory() as d:
        tp = Path(d) / "keywords.json"
        tp.write_text(
            '{"word_list":[{"name":"小云小云","threshold1":0.0,'
            '"subseg_threshold":-0.6,"cc_name":"mycc","extra":7}]}', encoding="utf-8")
        kj2 = build_keywords_json("你好小问", 0.5, template_path=str(tp))
        e = kj2["word_list"][0]
        check("模板覆盖 name", e["name"] == "你好小问", e["name"])
        check("模板覆盖 threshold1", abs(e["threshold1"] - 0.5) < 1e-9)
        check("模板保留 subseg_threshold", e["subseg_threshold"] == -0.6)
        check("模板保留自定义 cc_name", e["cc_name"] == "mycc")
        check("模板保留额外字段", e.get("extra") == 7)


def main() -> int:
    print("==== kws_train 自测 ====")
    test_config()
    test_early_stop()
    test_parse_score()
    test_summarize()
    test_threshold()
    test_keywords_json()
    print(f"\n==== 结果：PASS={_passed}  FAIL={_failed} ====")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
