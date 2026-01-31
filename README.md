# Audio to Sum of Sinusoids ✅

This small Python utility records audio from your microphone, computes the FFT to find the strongest frequency components, reconstructs the signal as a sum of sinusoids (top-N components), and plots:

- The amplitude spectrum (frequency vs amplitude)
- Original vs reconstructed time-domain waveform (first 50 ms)

Quick start:

1. Create a virtual environment and install deps:

   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt

2. Run (record 3 seconds and keep top 10 components):

   python audio_to_sinwaves.py --duration 3 --top 10

Live real-time plotting:

   python audio_to_sinwaves.py --live --window 4096 --hop 1024 --top 10

Transcription mode (dominant-note detection, suitable for piano single-note or dominant-note detection):

   python audio_to_sinwaves.py --transcribe --window 4096 --hop 1024 --top 10

Options for live/transcribe mode:
- `--window` — FFT window size in samples (default 4096)
- `--hop` — hop size in samples between updates (default 1024)
- `--transcribe` — use fast PyQtGraph UI with aubio pitch detector (if installed) to show live detected note

Notes:
- Make sure your microphone permissions are enabled.
- Lower `--hop` gives more frequent updates but uses more CPU; larger `--window` improves frequency resolution.
- `--transcribe` requires `pyqtgraph` and a Qt backend (e.g., `PyQt5`). `aubio` is optional but gives much more reliable pitch detection.

If you see an error like "Failed building wheel for aubio" on Windows, try one of these options:

- Easiest (recommended): install Miniconda/Anaconda and then:

  conda create -n audiotest -c conda-forge python=3.10 pyqt pyqtgraph aubio sounddevice numpy scipy matplotlib -y
  conda activate audiotest

- Try prebuilt wheel helper (`pipwin`):

  pip install pipwin
  pipwin install aubio

- Try pip (requires C build tools):

  Install Microsoft "Build Tools for Visual Studio" (C++ workload), then:
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install Cython numpy
  python -m pip install aubio

- Or skip aubio: the script will fall back to a simple strongest-peak heuristic and live plotting will still work.

- For convenience, `aubio` was moved to `optional-requirements.txt` (you can install it separately).

- Full polyphonic piano transcription requires specialized ML models (this shows dominant or prominent notes only).
- Use Ctrl-C or close the window to stop continuous live plotting.
