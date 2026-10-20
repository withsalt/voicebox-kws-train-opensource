# kws-train · 中文唤醒词 KWS：评估 + 训练 + 导出

一套精简、可独立运行的中文语音唤醒词（KWS, Keyword Spotting）工具，基于阿里
[`speech_charctc_kws_phone`](https://modelscope.cn/models/iic/speech_charctc_kws_phone)
（4 层 cFSMN / char-CTC 近场模型）做 finetune，包含三个模块：

| 模块 | 作用 | 命令 |
| --- | --- | --- |
| **kws_score** | 评估唤醒词质量（音节/误唤醒/可建模性等多维打分，并推荐近音正样本变体） | `python -m kws_score.cli <词>` |
| **kws_train** | 在基线模型上 finetune + 评测（输出 score.txt 与 DET，自动选阈值写 keywords.json） | `python -m kws_train train <词>` |
| **kws_export** | 把训练最优 checkpoint 导出为带 cache 的**流式 ONNX** + 部署资源 | `python -m kws_export export <词>` |

> 本仓库**不含数据集采集流程**：训练数据需你按下文「训练数据格式」自行准备（自录、公开语料、TTS 合成等任意来源）。
> `demo/` 提供一个纯麦克风的流式唤醒推理示例，演示如何使用导出的 ONNX。

---

## 预训练模型
唤醒词模型下载：  
[https://www.modelscope.cn/models/withsalt/voicebox_models/tree/master/v1/kws](https://www.modelscope.cn/models/withsalt/voicebox_models/tree/master/v1/kws)  

在线体验：  
[https://kws.pub.quarkbook.com/](https://kws.pub.quarkbook.com/)  

（PS：“小云小云”由仓库官方训练，其他唤醒词模型由当前仓库训练而来）  

## 安装

需要 Python ≥ 3.10。

```bash
# 1) 依赖
pip install -r requirements.txt

# 2) PyTorch：按你的 CUDA 版本从官方索引单独装（torch/torchaudio/torchvision 三者 CUDA 版本要对齐）
#    例（CUDA 12.x，按需替换 cuXXX）：
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3) 把 src/ 下各 kws_* 包注册为可导入（src 布局，首次必做）
pip install -e .

# 4) 准备配置
cp config.example.toml config.toml   # config.toml 已 gitignore
```

### 下载基线模型

训练与导出都依赖本地基线模型目录 `speech_charctc_kws_phone/`（约 148M）。从 ModelScope 下载到项目根：

```bash
# 方式一：git（需 git-lfs）
git clone https://www.modelscope.cn/iic/speech_charctc_kws_phone.git

# 方式二：modelscope CLI
modelscope download --model iic/speech_charctc_kws_phone --local_dir speech_charctc_kws_phone
```

下载后目录名应为 `speech_charctc_kws_phone`（与 `config.toml` 的 `[train].model_dir` 一致）。

---

## 步骤一：唤醒词评估（kws_score）

仅需评估侧依赖（`anthropic` + `pypinyin`）。在 `config.toml` 的 `[score]` 填好
`base_url` / `api_key` / `model`（或用环境变量 `KWS_SCORE_BASE_URL` / `KWS_SCORE_API_KEY` / `KWS_SCORE_MODEL`）。

```bash
# 给一个或多个候选唤醒词打分
python -m kws_score.cli 小白小白 你好小白

# 写出结构化 JSON
python -m kws_score.cli 小白小白 --json scores.json
```

输出包含多维评分（音节结构 / 音素区分度 / 误唤醒风险 / 发音清晰度 / 可建模性）、总分与等级，
并给出**近音正样本变体**推荐（仅声调/模糊音差异、排除完全同音的同长度近音词）——可作为训练正样本的扩充方向。
客观维度（音节数、OOV、拼音）本地确定性计算；误唤醒等需世界知识的维度调用大模型。

---

## 步骤二：准备训练数据

训练器吃 4 个文件，统一放在 `output/<唤醒词>/` 下（`<唤醒词>` 为目录安全名，叠词直接用原文，如 `output/小木小木/`）：

| 文件 | 格式（每行，`\t` 分隔） | 说明 |
| --- | --- | --- |
| `train_wav.scp` | `utt_id <TAB> /路径/到.wav` | 训练集音频清单 |
| `cv_wav.scp` | `utt_id <TAB> /路径/到.wav` | 交叉验证集（用于早停/选模，须与 train 不重叠） |
| `test_wav.scp` | `utt_id <TAB> /路径/到.wav` | 测试集（评测、画 DET、选阈值，须与 train/cv 不重叠） |
| `merge_trans.txt` | `utt_id <TAB> 空格分词转写` | 以上三个 scp 中**所有** utt 的转写（合并到一个文件） |

转写规则（char-CTC）：

- **正样本**（说了唤醒词）：转写 = 唤醒词**逐字空格分开**，如唤醒词「小木小木」→ `小 木 小 木`。
- **负样本**（未说唤醒词的真实语音）：转写 = 该段语音的**真实逐字转写**，如 `今 天 天 气 不 错`。
  负样本是压低误唤醒的关键，应覆盖日常语句、近音词、背景人声等。

音频要求：16kHz / 单声道 / 16-bit PCM WAV。`wav.scp` 中的路径建议用绝对路径（相对路径相对于运行目录解析）。

示例 `train_wav.scp`：

```
pos_0001	/data/kws/wav/pos_0001.wav
neg_0001	/data/kws/wav/neg_0001.wav
```

示例 `merge_trans.txt`（同一文件里正负混排）：

```
pos_0001	小 木 小 木
neg_0001	今 天 天 气 不 错
```

> 提示：先用 `python -m kws_train info 小木小木` 自检 4 个文件是否就绪、CUDA 是否可用。

---

## 步骤三：训练 finetune + 评测（kws_train）

需 GPU。读 `[train]` 配置，在基线模型上 finetune，默认 lr 已下调并带 cv_loss 早停；
训练后自动评测出 `score.txt` + DET，并按 `target_fa_per_hour` 误唤醒预算自动选阈值、回写部署 `keywords.json`。

```bash
# 默认参数训练（数据须已按步骤二备好）
python -m kws_train train 小木小木

# 覆盖部分参数
python -m kws_train train 小木小木 --epochs 20 --lr 5e-5 --patience 5
python -m kws_train train 小木小木 --no-eval     # 只训练不评测

# 查看配置 / 数据就绪 / GPU
python -m kws_train info 小木小木
```

产物落在 `output/小木小木/training/`（checkpoint、`score.txt`、DET 图、`keywords.json` 等）。

---

## 步骤四：导出流式 ONNX（kws_export）

把训练「最优解」checkpoint（默认 `avg_<num_avg>.pt`，否则回退 cv_loss 最小单 epoch）导出为
带显式 cache 的**流式 ONNX** + 部署资源（tokens / lexicon / am.mvn / `kws_meta.json`），供 C# 等运行时低延迟流式唤醒。
导出后默认跑 onnxruntime vs torch 的流式 parity 校验。需 `funasr + onnx + onnxruntime`（CPU 即可）。

```bash
python -m kws_export export 小木小木
python -m kws_export export 小木小木 --opset 17 --no-verify

# 查看导出配置 / 训练产物就绪 / 依赖可用性
python -m kws_export info 小木小木
```

产物落在 `output/小木小木/onnx/`（`kws_streaming.onnx`、`kws_meta.json`、`keywords.json` 等）。

---

## 推理 Demo（demo/）

`demo/` 是一个独立的麦克风流式唤醒示例（纯 `numpy` 前端 + `onnxruntime`），演示如何加载导出的 ONNX 做实时唤醒：

```bash
cd demo
pip install -r requirements.txt
python kws_demo.py            # 详见 demo/README.md
```

`demo/onnx/` 自带一份已导出的示例模型，可直接试跑；也可替换为 `kws_export` 导出的你自己的模型。

---

## 目录结构

```
.
├── config.example.toml      # 配置示例（[score]/[train]/[export]），复制为 config.toml 使用
├── requirements.txt         # 评估侧依赖
├── requirements-train.txt   # 训练 + 导出依赖（另需按 CUDA 装 torch）
├── pyproject.toml
├── src/
│   ├── kws_common/          # 共用小工具（safe_filename）
│   ├── kws_score/           # 唤醒词评估
│   ├── kws_train/           # 训练 + 评测
│   └── kws_export/          # 导出流式 ONNX
├── tests/                   # 单元测试
├── demo/                    # 麦克风流式唤醒推理示例
└── speech_charctc_kws_phone/   # 基线模型（自行从 ModelScope 下载，已 gitignore）
```

## 测试

```bash
# 评估侧（无需 GPU/torch；需 pypinyin）
PYTHONPATH=src python -m pytest tests/test_kws_score.py -q

# 训练/导出（需相应依赖与基线模型）
PYTHONPATH=src python -m pytest tests/test_kws_train.py tests/test_kws_export.py -q
```

## 致谢

- 基线模型：[iic/speech_charctc_kws_phone](https://modelscope.cn/models/iic/speech_charctc_kws_phone)（Apache-2.0）
- 训练框架：[ModelScope](https://github.com/modelscope/modelscope) / [FunASR](https://github.com/modelscope/FunASR)
