#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自定义唤醒词 · 实时麦克风 Demo（独立运行）

把训练导出的流式 ONNX 接到麦克风，实时检测唤醒词。唤醒词由模型决定（读 kws_meta.json 的
keyword 字段），不写死在代码里——换个模型目录即可换唤醒词。
零重依赖：只用 numpy + onnxruntime + sounddevice（不依赖 torch / funasr / 本仓库其它代码）。

用法：
    python kws_demo.py                       # 列出麦克风 → 选择 → 开始监听
    python kws_demo.py --list-devices        # 只列出麦克风设备后退出
    python kws_demo.py --device 1            # 跳过交互，直接用 1 号设备
    python kws_demo.py --onnx-dir ./onnx     # 指定模型资源目录（默认为当前目录下的 onnx/）
    python kws_demo.py --threshold 0.40      # 唤醒阈值（默认 0.25，越高越严）
    python kws_demo.py --cooldown 1.0        # 两次唤醒之间的最短间隔（秒）

对着麦克风连读唤醒词即触发 🔔；下方实时显示置信度条。Ctrl-C 退出。

注意：本 demo 的前端（kaldi fbank）与检测器是对训练/部署管线的**纯 Python 近似实现**，
置信度刻度可能与 Linux 版 kwsbp 略有差异；若现场触发偏难/偏易，用 --threshold 微调即可。
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from pathlib import Path

import numpy as np

def _enable_utf8_console():
    """Windows 控制台默认 GBK，输出 emoji/中文会抛 UnicodeEncodeError。尽量切到 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


DEFAULT_ONNX_DIR = "./onnx"   # 默认模型资源目录（相对当前工作目录）

LOG_EPSILON = 1.1920928955078125e-07  # torchaudio.compliance.kaldi 的 log 下限（与训练前端一致）


# ============================== 前端：纯 numpy kaldi fbank ==============================
# 逐项镜像 torchaudio.compliance.kaldi.fbank（funasr WavFrontend 同款配置）：
#   80 mel / 25ms 帧长 / 10ms 帧移 / hamming 窗 / dither=0 / snip_edges=True / 波形×32768。
# 训练导出的 CMVN(add_shift/rescale) 正是基于这套前端统计得到，故必须逐位对齐。

def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log(1.0 + f / 700.0)


def build_mel_filters(num_bins: int, fft_size: int, sample_rate: int,
                      low_freq: float = 20.0, high_freq: float = 0.0) -> np.ndarray:
    """构造 [num_bins, fft_size//2 + 1] 的三角 mel 滤波器组（与 kaldi/torchaudio 一致）。

    最后一列（nyquist）补 0，使其可直接与 rfft 功率谱（fft_size//2+1 点）相乘。
    """
    nyquist = sample_rate / 2.0
    if high_freq <= 0.0:
        high_freq = nyquist + high_freq  # high_freq=0 → nyquist(=8000)
    num_fft_bins = fft_size // 2                  # 256（不含 nyquist；nyquist 列后补 0）
    fft_bin_width = sample_rate / fft_size        # 31.25 Hz

    mel_low, mel_high = _hz_to_mel(np.array(low_freq)), _hz_to_mel(np.array(high_freq))
    delta = (mel_high - mel_low) / (num_bins + 1)

    # 每个 fft bin 的中心频率（i=0..255）对应的 mel
    fft_mel = _hz_to_mel(fft_bin_width * np.arange(num_fft_bins))   # [256]

    filters = np.zeros((num_bins, num_fft_bins), dtype=np.float64)
    for b in range(num_bins):
        left = mel_low + b * delta
        center = mel_low + (b + 1) * delta
        right = mel_low + (b + 2) * delta
        up = (fft_mel - left) / (center - left)
        down = (right - fft_mel) / (right - center)
        filters[b] = np.maximum(0.0, np.minimum(up, down))
    # 右补一列 0 对齐 rfft 输出的 nyquist 频点 → [num_bins, num_fft_bins+1]
    return np.pad(filters, ((0, 0), (0, 1)), mode="constant").astype(np.float32)


class FbankExtractor:
    """kaldi 兼容 fbank（snip_edges=True，帧长 400、帧移 160 @16k）。"""

    def __init__(self, cfg: dict, sample_rate: int = 16000):
        self.sr = sample_rate
        self.num_mel = int(cfg.get("num_mel_bins", 80))
        self.frame_len = int(round(cfg.get("frame_length_ms", 25) * sample_rate / 1000))   # 400
        self.frame_shift = int(round(cfg.get("frame_shift_ms", 10) * sample_rate / 1000))  # 160
        self.preemph = 0.97
        self.wav_scale = float(cfg.get("waveform_scale", 32768.0))
        self.fft_size = 1
        while self.fft_size < self.frame_len:
            self.fft_size *= 2                                                             # 512
        n = self.frame_len
        self.window = (0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))).astype(np.float64)
        self.mel = build_mel_filters(self.num_mel, self.fft_size, sample_rate).astype(np.float64)

    def num_frames(self, num_samples: int) -> int:
        if num_samples < self.frame_len:
            return 0
        return 1 + (num_samples - self.frame_len) // self.frame_shift

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        """samples: 1D float32（[-1,1] 量级）→ 返回 [n, num_mel] 的 log-mel（float32）。"""
        n = self.num_frames(len(samples))
        if n == 0:
            return np.empty((0, self.num_mel), dtype=np.float32)
        # 取帧矩阵 [n, frame_len]，转 int16 量级
        idx = np.arange(self.frame_len)[None, :] + self.frame_shift * np.arange(n)[:, None]
        frames = samples[idx].astype(np.float64) * self.wav_scale
        frames -= frames.mean(axis=1, keepdims=True)               # remove_dc_offset
        # 预加重：x[i]-=0.97*x[i-1]，x[0]-=0.97*x[0]
        pre = np.empty_like(frames)
        pre[:, 1:] = frames[:, 1:] - self.preemph * frames[:, :-1]
        pre[:, 0] = frames[:, 0] - self.preemph * frames[:, 0]
        pre *= self.window                                         # hamming
        spec = np.fft.rfft(pre, n=self.fft_size, axis=1)           # [n, fft//2+1]
        power = (spec.real ** 2 + spec.imag ** 2)                  # 功率谱
        mel_energy = power @ self.mel.T                            # [n, num_mel]
        return np.log(np.maximum(mel_energy, LOG_EPSILON)).astype(np.float32)


# ============================== 流式 KWS（前端→ONNX→检测，带跨块状态） ==============================

class StreamingKWS:
    """喂入 16k 单声道 float32 音频块，输出实时唤醒检测。所有状态跨块持有。"""

    def __init__(self, session, meta: dict, threshold: float, cooldown: float):
        self.sess = session
        self.fbank = FbankExtractor(meta["fbank"], meta.get("sample_rate", 16000))
        self.lfr_m = int(meta["lfr"]["m"])      # 5
        self.lfr_n = int(meta["lfr"]["n"])      # 3
        self.add_shift = np.asarray(meta["cmvn"]["add_shift"], dtype=np.float32)   # [400]
        self.rescale = np.asarray(meta["cmvn"]["rescale"], dtype=np.float32)       # [400]
        self.kw_tokens = list(meta["keyword_token_ids"])                           # [1462,1774,1462,1774]
        self.keyword = meta.get("keyword", "唤醒词")

        m = meta["model"]
        self.num_layers = int(m["num_layers"])
        proj, clen = int(m["proj_dim"]), int(m["cache_len"])
        self.caches = [np.zeros((1, proj, clen, 1), dtype=np.float32) for _ in range(self.num_layers)]
        self._cache_in = [f"cache{i}" for i in range(self.num_layers)]

        # 跨块缓冲
        self._sample_buf = np.zeros(0, dtype=np.float32)   # 帧重叠用
        self._fbank_buf = np.zeros((0, self.fbank.num_mel), dtype=np.float32)  # LFR 用
        self._lfr_started = False

        # 检测器：最近 W 帧的关键词 token 后验 [W, len(kw)]
        fps = self.fbank.sr / self.fbank.frame_shift / self.lfr_n     # ≈33.3
        self.win = max(int(round(meta.get("_max_dur", 2.5) * fps)), len(self.kw_tokens))
        self._post_buf = np.zeros((0, len(self.kw_tokens)), dtype=np.float64)

        self.threshold = float(threshold)
        self.cooldown = float(cooldown)
        self._last_fire = -1e9
        self.last_conf = 0.0
        self.fires = 0

    # ---- 前端：样本 → CMVN 后的 LFR 帧 [t,400] ----
    def _features(self, samples: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self._sample_buf, samples])
        nf = self.fbank.num_frames(len(buf))
        if nf == 0:
            self._sample_buf = buf
            return np.empty((0, self.add_shift.shape[0]), dtype=np.float32)
        fb = self.fbank(buf)                                   # [nf,80]
        # 保留下一块所需的样本重叠（下一帧起点 = nf*frame_shift）
        self._sample_buf = buf[nf * self.fbank.frame_shift:]

        # LFR：起始左填充首帧 ×(m-1)//2；之后窗口 m、步进 n
        frames = fb
        if not self._lfr_started and len(frames) > 0:
            pad = np.repeat(frames[:1], (self.lfr_m - 1) // 2, axis=0)
            frames = np.concatenate([pad, frames], axis=0)
            self._lfr_started = True
        buf_f = np.concatenate([self._fbank_buf, frames], axis=0)
        lfr_out = []
        pos = 0
        while pos + self.lfr_m <= len(buf_f):
            lfr_out.append(buf_f[pos:pos + self.lfr_m].reshape(-1))
            pos += self.lfr_n
        self._fbank_buf = buf_f[pos:]
        if not lfr_out:
            return np.empty((0, self.add_shift.shape[0]), dtype=np.float32)
        lfr = np.stack(lfr_out, axis=0).astype(np.float32)     # [t,400]
        return (lfr + self.add_shift) * self.rescale           # CMVN

    # ---- 检测器：in-order 峰值几何平均 ----
    def _update_detect(self, kw_post: np.ndarray, now: float) -> None:
        """kw_post: [t, len(kw)] 本批每帧的关键词 token 后验。逐帧更新滑窗并打分。"""
        for row in kw_post:
            self._post_buf = np.concatenate([self._post_buf, row[None, :]], axis=0)
            if len(self._post_buf) > self.win:
                self._post_buf = self._post_buf[-self.win:]
            # DP：best[i]=按时间顺序匹配前 i 个 token 的最大后验连乘（高→低更新，保证一帧只推进一个 token）
            k = len(self.kw_tokens)
            best = np.zeros(k + 1, dtype=np.float64)
            best[0] = 1.0
            for frame in self._post_buf:
                for i in range(k, 0, -1):
                    cand = best[i - 1] * frame[i - 1]
                    if cand > best[i]:
                        best[i] = cand
            conf = float(best[k] ** (1.0 / k))
            self.last_conf = conf
            if conf >= self.threshold and (now - self._last_fire) >= self.cooldown:
                self._last_fire = now
                self.fires += 1
                ts = time.strftime("%H:%M:%S")
                print(f"\r\033[K🔔  唤醒：{self.keyword}   置信度={conf:.3f}   [{ts}]   "
                      f"(第 {self.fires} 次)")
                # 触发后清空滑窗，避免同一次发声反复触发
                self._post_buf = np.zeros((0, k), dtype=np.float64)

    def process(self, samples: np.ndarray) -> None:
        feats = self._features(samples)
        if feats.shape[0] == 0:
            return
        ort_in = {"feats": feats[None, :, :].astype(np.float32)}
        for name, c in zip(self._cache_in, self.caches):
            ort_in[name] = c
        out = self.sess.run(None, ort_in)
        probs = out[0]                       # [1,t,2599]
        self.caches = list(out[1:])
        kw_post = probs[0][:, self.kw_tokens]   # [t, len(kw)]
        self._update_detect(kw_post, time.monotonic())


# ============================== 资源加载 ==============================

def load_runtime(threshold: float, onnx_dir: Path):
    """从 onnx_dir 加载模型与配置。需 kws_streaming.onnx + kws_meta.json；keywords.json 可选
    （仅用于取 max_dur，缺失则取默认 2.5s）。唤醒词、token、CMVN 等全部来自 kws_meta.json。"""
    import onnxruntime as ort  # 延迟导入，--list-devices 时不必装

    onnx_dir = Path(onnx_dir)
    onnx_path = onnx_dir / "kws_streaming.onnx"
    meta_path = onnx_dir / "kws_meta.json"
    kw_path = onnx_dir / "keywords.json"
    for p in (onnx_path, meta_path):
        if not p.exists():
            sys.exit(f"缺少模型资源：{p}\n请用 --onnx-dir 指定包含 kws_streaming.onnx 与 "
                     f"kws_meta.json 的目录（默认脚本同级 onnx/）。")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    max_dur = 2.5
    if kw_path.exists():
        try:
            entry = (json.loads(kw_path.read_text(encoding="utf-8")).get("word_list") or [{}])[0]
            max_dur = float(entry.get("max_dur", max_dur))
        except Exception:
            pass
    meta["_max_dur"] = max_dur

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    return sess, meta, float(threshold)


# ============================== 设备列举 / 选择 ==============================

def list_input_devices():
    import sounddevice as sd
    devices = sd.query_devices()
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = -1
    rows = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            try:
                ha = sd.query_hostapis(d["hostapi"])["name"]
            except Exception:
                ha = "?"
            rows.append((i, d["name"], ha, int(d.get("default_samplerate", 0)), i == default_in))
    return rows, default_in


def print_devices(rows):
    if not rows:
        print("  （未发现可用的麦克风输入设备）")
        return
    print("可用麦克风设备：")
    for i, name, ha, sr, is_def in rows:
        mark = "  ← 系统默认" if is_def else ""
        print(f"  [{i:>2}] {name}  | {ha} | {sr}Hz{mark}")


def choose_device(rows, default_in):
    print_devices(rows)
    if not rows:
        sys.exit("没有可用麦克风，无法启动。")
    valid = {r[0] for r in rows}
    while True:
        raw = input("\n请输入麦克风设备号（直接回车 = 系统默认）：").strip()
        if raw == "":
            return default_in if default_in in valid else rows[0][0]
        try:
            idx = int(raw)
        except ValueError:
            print("  请输入数字设备号。")
            continue
        if idx in valid:
            return idx
        print(f"  设备号 {idx} 不是有效的输入设备，请重新输入。")


# ============================== 线性重采样（设备不支持 16k 时回退用） ==============================

class LinearResampler:
    """流式线性插值重采样（用绝对样本索引保证跨块相位连续）。"""

    def __init__(self, sr_in: int, sr_out: int):
        self.step = sr_in / sr_out
        self.prev_tail = None
        self.t = 0.0       # 下一个输出点（输入样本单位，绝对坐标）
        self.in_pos = 0    # 当前块首样本的绝对索引

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.prev_tail is None:
            buf, base = x, 0
        else:
            buf = np.concatenate([self.prev_tail, x])
            base = self.in_pos - 1
        if len(buf) < 2:
            self.prev_tail = buf[-1:].copy() if len(buf) else self.prev_tail
            self.in_pos = base + len(buf)
            return np.zeros(0, dtype=np.float32)
        last_abs = base + len(buf) - 1
        ts = []
        t = self.t
        while t <= last_abs:
            ts.append(t)
            t += self.step
        if ts:
            xp = np.arange(base, base + len(buf))
            out = np.interp(np.array(ts), xp, buf).astype(np.float32)
            self.t = t
        else:
            out = np.zeros(0, dtype=np.float32)
        self.prev_tail = buf[-1:].copy()
        self.in_pos = base + len(buf)
        return out


# ============================== 主监听循环 ==============================

def run_listen(device: int, sess, meta, threshold: float, cooldown: float):
    import sounddevice as sd

    target_sr = int(meta.get("sample_rate", 16000))
    # 优先按 16k 打开；不支持则回退设备默认采样率 + 线性重采样
    open_sr = target_sr
    resampler = None
    try:
        sd.check_input_settings(device=device, samplerate=target_sr, channels=1, dtype="float32")
    except Exception:
        dev_sr = int(sd.query_devices(device)["default_samplerate"])
        open_sr = dev_sr
        resampler = LinearResampler(dev_sr, target_sr)
        print(f"⚠ 设备不支持 {target_sr}Hz，改用 {dev_sr}Hz 采集并实时重采样到 {target_sr}Hz。")

    kws = StreamingKWS(sess, meta, threshold, cooldown)
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            # 溢出/欠载只提示，不中断
            print(f"\r\033[K（音频状态：{status}）", end="", flush=True)
        q.put(indata[:, 0].copy())

    blocksize = max(int(open_sr * 0.03), 160)   # ~30ms
    dev_name = sd.query_devices(device)["name"]
    print(f"\n开始监听：设备[{device}] {dev_name}  采样率 {open_sr}Hz")
    print(f"唤醒词：『{kws.keyword}』   阈值：{threshold:.3f}   冷却：{cooldown:.1f}s")
    print("对着麦克风连读唤醒词试试；下方为实时置信度。按 Ctrl-C 退出。\n")

    bar_w = 30
    with sd.InputStream(device=device, channels=1, samplerate=open_sr, dtype="float32",
                        blocksize=blocksize, callback=callback):
        try:
            while True:
                chunk = q.get()
                if resampler is not None:
                    chunk = resampler.process(chunk)
                    if len(chunk) == 0:
                        continue
                kws.process(chunk)
                # 实时置信度条
                filled = int(min(kws.last_conf / max(threshold, 1e-6), 1.0) * bar_w)
                bar = "█" * filled + " " * (bar_w - filled)
                print(f"\r置信度 {kws.last_conf:0.3f} |{bar}| 阈值 {threshold:.3f}",
                      end="", flush=True)
        except KeyboardInterrupt:
            print(f"\n\n已退出。本次共唤醒 {kws.fires} 次。")


# ============================== CLI ==============================

def main():
    _enable_utf8_console()
    ap = argparse.ArgumentParser(
        description="自定义唤醒词 · 实时麦克风 Demo（ONNX，独立运行）")
    ap.add_argument("--list-devices", action="store_true", help="只列出麦克风设备后退出")
    ap.add_argument("--device", type=int, default=None, help="指定麦克风设备号（跳过交互选择）")
    ap.add_argument("--onnx-dir", default=DEFAULT_ONNX_DIR,
                    help="模型资源目录（含 kws_streaming.onnx / kws_meta.json），默认 .\\onnx（相对当前目录）")
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="唤醒阈值（默认 0.25，越高越严、越不易误唤醒）")
    ap.add_argument("--cooldown", type=float, default=1.0, help="两次唤醒最短间隔秒数（默认 1.0）")
    args = ap.parse_args()

    if args.list_devices:
        try:
            rows, _ = list_input_devices()
        except Exception as e:
            sys.exit(f"无法列举音频设备（请确认已安装 sounddevice 及其 PortAudio 运行库）：{e}")
        print_devices(rows)
        return

    sess, meta, threshold = load_runtime(args.threshold, Path(args.onnx_dir))

    try:
        import sounddevice  # noqa: F401
    except Exception as e:
        sys.exit(f"未安装 sounddevice（麦克风采集库）：{e}\n请先 `pip install -r requirements.txt`。")

    if args.device is not None:
        device = args.device
    else:
        rows, default_in = list_input_devices()
        device = choose_device(rows, default_in)

    run_listen(device, sess, meta, threshold, args.cooldown)


if __name__ == "__main__":
    main()
