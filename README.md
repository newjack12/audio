

### 📄 README.md

````markdown
# Melody App (Industrial Prototype)

This project is an **industrial-style prototype** for automatic **melody and chord transcription**
from audio signals (human humming/voice, string instruments, piano).

⚠️ **Important note**  
This project was **partially developed with the assistance of AI tools** (for architecture design,
algorithm integration, and code generation).  
It is **not a finished product** and is still **under active improvement**.

---

## 🎯 Project Objectives

- Extract **main melody** from:
  - Human voice / humming
  - String instruments (violin, guitar – monophonic)
  - Piano (polyphonic)
- Export results in **musically usable formats**
- Provide a **quick audio preview** to validate transcription quality

---

## ✨ Main Features

- 🎼 **Melody extraction**
  - Output as MIDI (`.mid`)
  - Note names and numeric notation (`.json`, `.txt`)
- 🎹 **Optional chord estimation**
  - Template-based chords
  - Omnizart-based chords (if available)
- 🔊 **Preview audio**
  - Original audio mixed with synthesized melody
- 🔄 **Automatic backend selection**
  - Uses the best available engine depending on installed libraries

---

## 📁 Output Files

After running the program, you may obtain:

- `02_melody.mid` – main melody (single track)
- `02_melody.json / txt` – structured note information
- `03_preview_mix.wav` – audio preview (original + melody)
- `04_chords.mid / csv` – optional chord transcription

---

## 🛠 Environment

- OS: **Windows**
- Python: **Conda environment**
- Audio processing via **ffmpeg**

---

## 🚀 Installation

Activate your environment:

```bash
conda activate basicpitch
````

Install dependencies:

```bash
python -m pip install -r requirements_min.txt
```

Optional (stronger models):

```bash
pip install crepe librosa torch piano_transcription_inference omnizart
```

> All advanced backends are **optional**.
> The system will automatically fall back if a dependency is missing.

---

## ▶️ Usage

Basic command:

```bash
python -m melody_app.cli input.wav --out out_dir --mode voice --engine auto
```

Examples:

**Voice / humming**

```bash
python -m melody_app.cli test.m4a --mode voice --engine auto
```

**String instruments**

```bash
python -m melody_app.cli input.wav --mode string
```

**Piano**

```bash
python -m melody_app.cli piano.wav --mode piano --device cpu
```

---

## 🚧 Current Limitations

* Chord detection accuracy is limited
* No graphical interface
* Performance depends on audio quality
* Models are not fine-tuned for all instruments

---


