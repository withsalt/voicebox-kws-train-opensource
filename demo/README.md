# 小木小木 唤醒词 · 实时麦克风 Demo

一个**独立、跨平台**的语音唤醒体验程序：对着麦克风说出唤醒词即可触发提示
（唤醒词由模型决定，程序启动时也会打印）。
内部用训练导出的流式 ONNX 模型（约 3MB，CPU 实时运行），无需联网、无需 GPU。

> 适用环境：Windows / macOS / Linux，Python 3.8 及以上。采集音频为 16kHz 单声道（程序自动处理）。

---

## 一、安装

```bash
# 1) 建议用独立虚拟环境（可选）
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 2) 安装依赖
pip install -r requirements.txt
```

> Linux 若提示找不到 PortAudio，需先装系统库：`sudo apt-get install libportaudio2`。
> Windows / macOS 的 `sounddevice` 轮子已自带 PortAudio，无需额外安装。

## 二、运行

请**在 demo 目录下运行**（默认从当前目录的 `.\onnx` 加载模型；放别处可用 `--onnx-dir` 指定）：

```bash
cd demo
python kws_demo.py
```

启动后会**列出所有麦克风设备**，输入对应编号回车即可（直接回车=系统默认麦克风）：

```
可用麦克风设备：
  [ 1] 麦克风 (Realtek Audio)        | Windows WASAPI | 16000Hz  ← 系统默认
  [ 2] 蓝牙耳机 (Hands-Free)         | Windows WASAPI | 16000Hz

请输入麦克风设备号（直接回车 = 系统默认）：
```

选好后进入实时监听。程序会先打印当前唤醒词，对着麦克风**连读唤醒词**（本包为「小木小木」，
一口气说完、中间不停顿效果最好）：

```
置信度 0.812 |████████████████████████████  | 阈值 0.250
🔔  唤醒：小木小木   置信度=0.812   [15:42:07]   (第 1 次)
```

底部一行是**实时置信度条**：说话时会跳动，喊出唤醒词时明显冲高；越过阈值即触发 🔔。
按 **Ctrl-C** 退出。

## 三、常用参数

| 参数 | 说明 |
|---|---|
| `--list-devices` | 只列出麦克风设备后退出 |
| `--device N` | 直接指定 N 号麦克风，跳过交互选择 |
| `--onnx-dir DIR` | 模型资源目录（含 `kws_streaming.onnx` / `kws_meta.json`），默认 `.\onnx`（相对当前目录，故请在 demo 目录内运行） |
| `--threshold X` | 唤醒阈值（默认 `0.25`，越高越严、越不易误唤醒） |
| `--cooldown S` | 两次唤醒之间的最短间隔秒数（默认 1.0） |

示例：

```bash
python kws_demo.py --list-devices
python kws_demo.py --device 2 --threshold 0.50
```

## 四、阈值怎么调？

- **阈值越高** → 越不容易误唤醒，但要求发音更标准（召回略降）；
- **阈值越低** → 越灵敏、越容易唤醒，但误唤醒风险升高。

默认 `0.25` 偏灵敏（便于现场体验快速唤醒）。说出唤醒词时置信度通常会冲到 0.8~1.0，
普通说话一般很低，所以现场体验时若觉得：

- **偶尔被普通说话误唤醒** → 调高，例如 `--threshold 0.45` 或 `0.60`；
- **不容易唤醒** → 调低，例如 `--threshold 0.20`。

建议在 **0.25 ~ 0.60** 之间按实际麦克风/环境微调。

## 五、目录结构

```
demo/
  kws_demo.py        # 主程序（单文件，纯 numpy 前端 + onnxruntime 推理）
  requirements.txt   # 依赖：numpy / onnxruntime / sounddevice
  README.md          # 本说明
  onnx/              # 模型与配置（自带，无需改动；可用 --onnx-dir 指向其它目录）
    kws_streaming.onnx   # 流式唤醒模型
    kws_meta.json        # 唤醒词、特征/模型参数（keyword、fbank、LFR、CMVN、token id 等）
    keywords.json        # 部署参数（可选）
```

> 唤醒词由模型决定（`onnx/kws_meta.json` 的 `keyword` 字段），不写死在程序里——
> 换一套 `onnx/` 目录（或 `--onnx-dir` 指过去）即可换唤醒词。**本包内置模型的唤醒词为「小木小木」**
> （程序启动时也会打印当前唤醒词）。

## 六、说明与边界

- 程序所有参数（唤醒词、fbank、LFR、CMVN、关键词 token 等）均从 `onnx/` 下的配置读取，
  不在代码里写死；更换唤醒词只需替换 `onnx/` 内的模型与配置（或用 `--onnx-dir` 指过去）。
- 本 Demo 的前端（kaldi fbank）与检测逻辑是对训练/部署管线的**纯 Python 近似实现**，
  便于跨平台体验；其置信度刻度与生产端（Linux 版 kwsbp / C# 运行时）可能存在细微差异，
  如刻度整体偏移，以现场麦克风实测、用 `--threshold` 校准为准。
- 唤醒效果受麦克风质量、距离、环境噪声影响。安静近场、发音清晰时效果最佳。
