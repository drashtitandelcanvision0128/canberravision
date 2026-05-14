# Canberra Vision AI (YOLO26)

Computer vision stack for **vehicle detection** and **PPE / safety gear detection** (helmet, vest, mask), with a **Gradio** web UI. Built around **Ultralytics YOLO** and related tooling (OCR, video, webcam).

## Requirements

- Python 3.10+ recommended  
- CUDA optional (GPU); CPU runs with reduced throughput  
- [Git](https://git-scm.com/) and [pip](https://pip.pypa.io/)

## Setup

```bash
cd YOLO26
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

> **Note:** `requirements.txt` pins **Gradio 5.25** and several heavy deps (PyTorch, PaddleOCR, etc.). Install time and disk use can be large.

## Run the main app

```bash
python apps/app.py
```

Then open the URL shown in the terminal (default **http://127.0.0.1:7860** unless overridden by env vars).

The UI includes **Vehicle detection** and **PPE detection** modes (image, video, webcam), plus model selection where configured.

## Other entry points (optional)

| Purpose | Command |
|--------|---------|
| Interactive launcher menu | `python QUICK_START.py` |
| Alternate / specialised Gradio apps | See `apps/` (e.g. plate-focused demos if present) |

Windows helpers: `start.bat`, `start_fast.bat`, `start_clean.bat` in the repo root.

## Repository layout

| Path | Role |
|------|------|
| `apps/` | Main applications (`app.py` and other Gradio / demos) |
| `src/` | Core detection, OCR, processors |
| `modules/` | Optional features (e.g. vehicle classification / DB helpers) |
| `models/` | Model weights and assets (as used by your setup) |
| `database/` | Local DB files (e.g. vehicle data) where applicable |
| `tools/` | Diagnostics and utilities |
| `docs/` | Extra documentation (if present) |

## Environment (deployment)

Common variables for hosted runs:

- `APP_ENV` — e.g. `production`  
- `GRADIO_SERVER_NAME` / `GRADIO_SERVER_PORT` — bind address and port  

See your deployment notes (Docker / Coolify / etc.) for full lists.

## Troubleshooting

1. Run system checks if available: `python tools/system_test.py`  
2. GPU: ensure PyTorch build matches your CUDA version, or use CPU-only wheels  
3. If the web UI misbehaves after dependency upgrades, align versions with `requirements.txt` and reinstall in a clean venv  

## License and upstream

Ultralytics and other upstream packages ship their own licenses (e.g. Ultralytics AGPL-3.0 where applicable). Check each dependency and model license for commercial use.

---

**Canberra Vision** — [canberravision.com](https://canberravision.com/)
