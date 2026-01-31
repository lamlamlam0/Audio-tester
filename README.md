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

Notes:
- Make sure your microphone permissions are enabled.
- Adjust `--fs`, `--duration`, and `--maxfreq` as needed.
