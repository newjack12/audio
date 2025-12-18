# melody_app_industrial

> 工业级旋律/和声转写（人声哼唱、弦乐器、钢琴），可在 Windows + conda 环境运行。
ffmpeg -i test-hu2.m4a -ac 1 -ar 44100 test-hu2.wav
## 你会得到什么

- **02_melody.mid**：主旋律 MIDI（单轨，适合直接进 DAW/MuseScore）
- **02_melody.json / txt**：可读结构化结果 + 音名 + 简谱
- **03_preview_mix.wav**：原音频 + 旋律合成的快速试听（不用 DAW 就能听对不对）
- **04_chords.mid / 04_chords.csv**：可选输出和弦（模板基线或 Omnizart）

## 安装（与你的 conda 环境兼容）

1) 进入你的环境：

```bash
conda activate basicpitch
```

2) 确保安装 ffmpeg（你环境里已经有）：

```bash
conda install -c conda-forge ffmpeg
```

3) 安装必要依赖：

```bash
python -m pip install -U pip
python -m pip install -r requirements_min.txt
```

4) （推荐）安装更强的后端（可选）：

```bash
# 人声/哼唱的 SOTA 单音高跟踪（更稳）
python -m pip install crepe

# pYIN 作为传统强基线
python -m pip install librosa

# 钢琴转写更强（Onsets & Frames 类系）
python -m pip install torch
python -m pip install piano_transcription_inference

# 和弦转写（可选，若依赖冲突可跳过，使用 template 版本即可）
python -m pip install omnizart
```

> 说明：所有后端都是 **可选的**，项目会自动选择已安装的最优后端；缺哪个就自动降级。

## 运行

在项目根目录（README 所在目录）执行：

```bash
python -m melody_app.cli <你的音频文件> --out out_dir --mode voice --engine auto --chords template

```

### 常用场景

- **人声/哼唱（推荐）**：

```bash
python -m melody_app.cli test-si1.m4a --mode voice --engine auto --chords none --out out_si1auto
 python -m melody_app.cli test-si1.m4a --mode voice --engine pyin --out out_si1


```

- **弦乐器（小提琴/吉他单旋律等）**：

```bash
python -m melody_app.cli input.wav --mode string --engine auto
```

- **钢琴（更强的钢琴模型）**：

```bash
python -m melody_app.cli piano.wav --mode piano --engine auto --device cpu --chords template
```

- **强制使用 Basic Pitch（通用多音符转写）**：

```bash
python -m melody_app.cli input.wav --mode poly --engine basic_pitch
```

- **输出和弦（Omnizart，若已安装）**：

```bash
python -m melody_app.cli input.wav --mode poly --engine basic_pitch --chords omnizart
```

## 输出目录说明

- `00_preprocessed_44100.wav`：统一预处理后的音频（用于试听/稳定推理）
- `01_raw_transcription.mid`：多音符转写结果（Basic Pitch / piano_transcription_inference）
- `01_pitch_track.csv`：单音高轨迹（CREPE/pYIN 路径才会有）
- `02_melody.*`：最终主旋律
- `03_preview_*`：试听
- `04_chords.*`：和声/和弦（可选）

