#!/usr/bin/env python3
"""
Real-time Spectrum Analyzer

Reads audio from the default microphone (laptop mic) and displays a live amplitude
spectrum. Uses `pyqtgraph` for low-latency plotting when available, and falls back
to `matplotlib` otherwise.

Usage:
  python audio_to_spectrum.py --backend auto --window 4096 --hop 512

Options:
  --backend  'auto'|'pyqt'|'matplotlib'
  --window   FFT window size (samples)
  --hop      hop / blocksize (samples)
  --fs       sampling rate (Hz)
  --maxfreq  maximum frequency to display (Hz)
  --device   sounddevice input device (optional)
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import sounddevice as sd
from scipy.signal import get_window, find_peaks

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore
    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


def _parabolic_interp(mags, idx):
    """Parabolic interpolation around index idx. Return (idx + delta, peak_db).
    Works on linear magnitude by converting to dB internally."""
    N = len(mags)
    if idx <= 0 or idx >= N - 1:
        # no neighbors
        mag_db = 20 * np.log10(max(mags[idx], 1e-12))
        return float(idx), mag_db
    m1 = 20 * np.log10(max(mags[idx - 1], 1e-12))
    m2 = 20 * np.log10(max(mags[idx], 1e-12))
    m3 = 20 * np.log10(max(mags[idx + 1], 1e-12))
    denom = (m1 - 2 * m2 + m3)
    if denom == 0:
        delta = 0.0
    else:
        delta = 0.5 * (m1 - m3) / denom
    peak_db = m2 - 0.25 * (m1 - m3) * delta
    return float(idx) + float(delta), peak_db

class RealTimeSpectrum:
    def __init__(self, fs=44100, window_size=4096, hop_size=512, min_freq=27.5, max_freq=4186.0, smooth=4, log_scale=True, min_amp=0.001, peak_count=6, peak_thresh=0.05, device=None, backend='auto'):
        self.fs = int(fs)
        self.window_size = int(window_size)
        self.hop_size = int(hop_size)
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.smooth_k = int(smooth) if int(smooth) >= 1 else 1
        self.log_scale = bool(log_scale)
        self.min_amp = float(min_amp)
        self.peak_count = int(peak_count)
        self.peak_thresh = float(peak_thresh)
        self.device = device

        # buffer & window (Hann window)
        self.buffer = np.zeros(self.window_size, dtype='float32')
        self.window = get_window('hann', self.window_size).astype('float32')
        self.freqs = np.fft.rfftfreq(self.window_size, 1.0 / self.fs)
        # mask frequencies in the requested piano range (A0..C8)
        self.mask = (self.freqs >= self.min_freq) & (self.freqs <= self.max_freq)

        # spectrogram history
        self.history_len = 400
        self.spec_history = np.zeros((self.mask.sum(), self.history_len), dtype='float32')
        # smoothing buffer for temporal smoothing of the spectrum
        self.smooth_hist = np.zeros((self.mask.sum(), self.smooth_k), dtype='float32')
        self.smooth_idx = 0
        self.smooth_initialized = False
        # debug flag (set True to print update timestamps)
        self.debug = False

        # choose backend
        if backend == 'auto':
            if PYQT_AVAILABLE:
                self.backend = 'pyqt'
            elif MATPLOTLIB_AVAILABLE:
                self.backend = 'matplotlib'
            else:
                raise RuntimeError('No plotting backend available. Install pyqtgraph or matplotlib')
        else:
            self.backend = backend

        if self.backend == 'pyqt' and not PYQT_AVAILABLE:
            raise RuntimeError("pyqtgraph not available. Install 'pyqtgraph' and a Qt binding (PyQt5/PySide2)")
        if self.backend == 'matplotlib' and not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("matplotlib not available. Install 'matplotlib'")

        self._init_plot()

        # sounddevice input stream
        self.stream = sd.InputStream(callback=self._audio_callback,
                                     channels=1,
                                     samplerate=self.fs,
                                     blocksize=self.hop_size,
                                     dtype='float32',
                                     device=self.device)

    def _init_plot(self):
        if self.backend == 'pyqt':
            pg.setConfigOptions(antialias=False)
            self.app = pg.mkQApp('Realtime Spectrum')
            self.win = pg.GraphicsLayoutWidget(title='Realtime Spectrum Analyzer')
            self.win.resize(1000, 600)

            # spectrum
            self.spec_plot = self.win.addPlot(row=0, col=0, title='Spectrum')
            self.spec_plot.setLabel('bottom', 'Frequency', units='Hz')
            self.spec_plot.setLabel('left', 'Amplitude')
            self.spec_curve = self.spec_plot.plot(self.freqs[self.mask], np.zeros(self.mask.sum()), pen='c')
            self.spec_plot.setXRange(self.min_freq, self.max_freq)
            # peaks scatter and labels
            self.peaks_scatter = pg.ScatterPlotItem(size=10, brush='r')
            self.spec_plot.addItem(self.peaks_scatter)
            self.peak_texts = [pg.TextItem(anchor=(0.5, 0)) for _ in range(self.peak_count)]
            for t in self.peak_texts:
                self.spec_plot.addItem(t)

            # spectrogram (image)
            self.win.nextRow()
            self.img_view = self.win.addViewBox(row=1, col=0)
            self.img = pg.ImageItem()
            self.img_view.addItem(self.img)
            self.img_view.setAspectLocked(True)
            # optionally set Y range to show our freq range (image is freq x time)
            try:
                self.img_view.setYRange(self.min_freq, self.max_freq)
            except Exception:
                pass

            # configure timer
            self.timer = QtCore.QTimer()
            interval_ms = max(10, int(1000.0 * self.hop_size / self.fs))
            self.timer.setInterval(interval_ms)
            self.timer.timeout.connect(self.update)

        else:
            self.fig, (self.ax_spec, self.ax_specgram) = plt.subplots(2, 1, figsize=(10, 6))
            self.line, = self.ax_spec.plot(self.freqs[self.mask], np.zeros(self.mask.sum()))
            # peak markers (matplotlib)
            self.peak_marker, = self.ax_spec.plot([], [], 'ro', markersize=6)
            self.peak_annots = []
            # use log scale for x-axis if requested
            if self.log_scale:
                try:
                    self.ax_spec.set_xscale('log')
                except Exception:
                    pass
            self.ax_spec.set_xlim(self.min_freq, self.max_freq)
            self.ax_spec.set_xlabel('Frequency (Hz)')
            self.ax_spec.set_ylabel('Amplitude')

            # spectrogram image
            img = np.zeros((self.mask.sum(), self.history_len))
            self.im = self.ax_specgram.imshow(img, aspect='auto', origin='lower',
                                             extent=[-self.history_len, 0, self.min_freq, self.max_freq], cmap='inferno')
            self.ax_specgram.set_ylabel('Frequency (Hz)')
            self.ax_specgram.set_xlabel('Time (frames)')

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        data = indata[:, 0]
        self.buffer = np.roll(self.buffer, -len(data))
        self.buffer[-len(data):] = data

    def start(self):
        self.stream.start()
        try:
            if self.backend == 'pyqt':
                print('Starting realtime spectrum (pyqtgraph). Close window to stop.')
                self.win.show()
                self.timer.start()
                self.app.exec_()
            else:
                print('Starting realtime spectrum (matplotlib). Close window to stop.')
                interval_ms = max(10, int(1000.0 * self.hop_size / self.fs))
                self.ani = FuncAnimation(self.fig, self._matplotlib_update, interval=interval_ms, blit=False)
                plt.show()
        finally:
            self.stream.stop()

    def update(self):
        # compute FFT
        if getattr(self, 'debug', False):
            import time
            print('update', time.time())
        w = self.buffer * self.window
        X = np.fft.rfft(w)
        mags = np.abs(X) / self.window_size
        if len(mags) > 2:
            mags[1:-1] *= 2
        mags = mags[self.mask]

        # temporal smoothing (moving average across last self.smooth_k frames)
        if self.smooth_k > 1:
            if not self.smooth_initialized:
                # initialize with current spectrum to avoid startup ramp
                self.smooth_hist[:, :] = mags[:, None]
                self.smooth_initialized = True
                self.smooth_idx = 1 % self.smooth_k
            else:
                self.smooth_hist[:, self.smooth_idx] = mags
                self.smooth_idx = (self.smooth_idx + 1) % self.smooth_k
            mags_smoothed = self.smooth_hist.mean(axis=1)
        else:
            mags_smoothed = mags

        # clamp to minimum amplitude to avoid plotting/DB issues
        mags_smoothed = np.maximum(mags_smoothed, self.min_amp)

        # update spectrum (use smoothed values)
        if self.backend == 'pyqt':
            self.spec_curve.setData(self.freqs[self.mask], mags_smoothed)
            try:
                maxv = max(np.max(mags_smoothed) * 1.1 + 1e-12, self.min_amp * 1.001)
                self.spec_plot.setYRange(self.min_amp, maxv)
            except Exception:
                pass
        else:
            self.line.set_ydata(mags_smoothed)
            maxv = max(np.max(mags_smoothed) * 1.1 + 1e-12, self.min_amp * 1.001)
            self.ax_spec.set_ylim(self.min_amp, maxv)

        # update spectrogram history (convert to dB and normalize)
        mags_clipped = np.maximum(mags_smoothed, self.min_amp)
        mags_db = 20 * np.log10(mags_clipped)
        vmin, vmax = -100.0, 0.0
        col = (mags_db - vmin) / (vmax - vmin)
        col = np.clip(col, 0.0, 1.0)
        self.spec_history = np.roll(self.spec_history, -1, axis=1)
        self.spec_history[:, -1] = col

        if self.backend == 'pyqt':
            # image expects 2D array (freq x time)
            self.img.setImage(self.spec_history, autoLevels=False)
        else:
            self.im.set_data(self.spec_history)

        # Peak detection and annotation
        try:
            masked_freqs = self.freqs[self.mask]
            thresh = np.max(mags_smoothed) * self.peak_thresh if np.max(mags_smoothed) > 0 else self.min_amp
            peaks, props = find_peaks(mags_smoothed, height=thresh, distance=1)
            if len(peaks) > 0:
                peak_vals = mags_smoothed[peaks]
                order = np.argsort(peak_vals)[-self.peak_count:][::-1]
                sel = peaks[order]
                df = masked_freqs[1] - masked_freqs[0] if masked_freqs.size > 1 else 0.0

                freqs_sel = []
                mags_sel = []
                ann_texts = []
                for i in sel:
                    idx_interp, peak_db = _parabolic_interp(mags_smoothed, int(i))
                    freq = masked_freqs[0] + idx_interp * df
                    freqs_sel.append(freq)
                    mags_sel.append(mags_smoothed[int(round(i))])
                    ann_texts.append(f"{freq:.1f} Hz\n{peak_db:.1f} dB")

                if self.backend == 'pyqt':
                    spots = [{'pos': (f, m)} for f, m in zip(freqs_sel, mags_sel)]
                    self.peaks_scatter.setData(spots)
                    for t, txt, f, m in zip(self.peak_texts, ann_texts, freqs_sel, mags_sel):
                        t.setHtml(f"<div style='color:#ff3333;font-size:10pt'>{txt}</div>")
                        t.setPos(f, m)
                else:
                    self.peak_marker.set_data(freqs_sel, mags_sel)
                    for a in self.peak_annots:
                        a.remove()
                    self.peak_annots = []
                    for f, m, txt in zip(freqs_sel, mags_sel, ann_texts):
                        a = self.ax_spec.text(f, m, txt, fontsize=8, ha='center', va='bottom', color='red')
                        self.peak_annots.append(a)
            else:
                if self.backend == 'pyqt':
                    self.peaks_scatter.setData([])
                    for t in self.peak_texts:
                        t.setHtml("")
                else:
                    self.peak_marker.set_data([], [])
                    for a in self.peak_annots:
                        a.remove()
                    self.peak_annots = []
        except Exception:
            # don't let peak annotation errors break the main loop
            pass

    def _matplotlib_update(self, frame):
        self.update()
        # force a redraw to ensure the UI updates in all backends
        try:
            self.fig.canvas.draw_idle()
            import matplotlib.pyplot as _plt
            _plt.pause(0.001)
        except Exception:
            pass
        return self.line, self.im


def main():
    p = argparse.ArgumentParser(description='Real-time Spectrum Analyzer')
    p.add_argument('--backend', choices=['auto', 'pyqt', 'matplotlib'], default='auto')
    p.add_argument('--window', type=int, default=4096)
    p.add_argument('--hop', type=int, default=512)
    p.add_argument('--fs', type=int, default=44100)
    p.add_argument('--minfreq', type=float, default=27.5, help='Minimum frequency to display (Hz). Default A0=27.5 Hz')
    p.add_argument('--maxfreq', type=float, default=4186.0, help='Maximum frequency to display (Hz). Default C8=4186 Hz')
    p.add_argument('--smooth', type=int, default=4, help='Temporal smoothing window (frames). Default 4')
    p.add_argument('--minamp', type=float, default=0.001, help='Minimum amplitude displayed (linear). Default 0.001')
    p.add_argument('--peakcount', type=int, default=6, help='Number of peaks to show (default 6)')
    p.add_argument('--peakthresh', type=float, default=0.05, help='Relative peak detection threshold (fraction of max). Default 0.05')
    p.add_argument('--no-log', action='store_true', help='Disable logarithmic frequency axis')
    p.add_argument('--device', default=None)
    p.add_argument('--debug', action='store_true', default=False, help='print update timestamps')
    args = p.parse_args()

    try:
        analyzer = RealTimeSpectrum(fs=args.fs, window_size=args.window, hop_size=args.hop,
                                    min_freq=args.minfreq, max_freq=args.maxfreq, smooth=args.smooth, log_scale=not args.no_log, min_amp=args.minamp, peak_count=args.peakcount, peak_thresh=args.peakthresh, device=args.device, backend=args.backend)
        analyzer.debug = args.debug
        analyzer.start()
    except Exception as e:
        print('Error:', e, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()


def analyze_and_reconstruct(x, fs, top_n=10, max_freq=None):
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, 1.0 / fs)

    # Magnitude scaling (so amplitudes reflect sinusoid amplitudes)
    mags = np.abs(X) / N
    if N > 1:
        mags[1:-1] *= 2

    # Limit to max_freq if requested
    if max_freq is not None:
        mask = freqs <= max_freq
        freqs_limited = freqs[mask]
        mags_limited = mags[mask]
        X_limited = X[mask]
    else:
        freqs_limited = freqs
        mags_limited = mags
        X_limited = X

    # Pick top N peaks (by magnitude)
    # Use simple argpartition to get top magnitudes
    idx = np.argsort(mags_limited)[-top_n:][::-1]
    top_freqs = freqs_limited[idx]
    top_mags = mags_limited[idx]
    top_phases = np.angle(X_limited[idx])

    # Sort by frequency for nicer plotting and synthesis
    order = np.argsort(top_freqs)
    top_freqs = top_freqs[order]
    top_mags = top_mags[order]
    top_phases = top_phases[order]

    # Reconstruct signal as sum of sinusoids
    t = np.arange(N) / fs
    recon = np.zeros_like(t)
    for A, f, phi in zip(top_mags, top_freqs, top_phases):
        recon += A * np.cos(2 * np.pi * f * t + phi)

    return {
        'freqs': freqs,
        'mags': mags,
        'top_freqs': top_freqs,
        'top_mags': top_mags,
        'top_phases': top_phases,
        'recon': recon,
        't': t,
    }




def _analyze_window_and_get_top(window, fs, top_n=10, max_freq=None):
    # Keep backward-compatible helper for single-window analysis
    N = len(window)
    X = np.fft.rfft(window)
    freqs = np.fft.rfftfreq(N, 1.0 / fs)

    mags = np.abs(X) / N
    if N > 1:
        mags[1:-1] *= 2

    if max_freq is not None:
        mask = freqs <= max_freq
        freqs_limited = freqs[mask]
        mags_limited = mags[mask]
        X_limited = X[mask]
    else:
        freqs_limited = freqs
        mags_limited = mags
        X_limited = X

    idx = np.argsort(mags_limited)[-top_n:][::-1]
    top_freqs = freqs_limited[idx]
    top_mags = mags_limited[idx]
    top_phases = np.angle(X_limited[idx])

    order = np.argsort(top_freqs)
    top_freqs = top_freqs[order]
    top_mags = top_mags[order]
    top_phases = top_phases[order]

    t = np.arange(N) / fs
    recon = np.zeros_like(t)
    for A, f, phi in zip(top_mags, top_freqs, top_phases):
        recon += A * np.cos(2 * np.pi * f * t + phi)

    return {
        'freqs': freqs,
        'mags': mags,
        'top_freqs': top_freqs,
        'top_mags': top_mags,
        'top_phases': top_phases,
        'recon': recon,
        't': t,
    }


# Try to import aubio and pyqtgraph for real-time transcription/plotting
try:
    import aubio
    aubio_available = True
except Exception:
    aubio_available = False

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui
    pyqt_available = True
except Exception:
    pyqt_available = False


NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def freq_to_note_name(freq):
    if freq <= 0 or np.isnan(freq):
        return None
    # MIDI note number
    midi = int(round(69 + 12 * np.log2(freq / 440.0)))
    octave = midi // 12 - 1
    name = NOTE_NAMES[midi % 12]
    return f"{name}{octave}"


class LiveTranscriber:
    """Fast live plotting and simple transcription optimized for piano notes.

    Notes:
    - Uses pyqtgraph for fast plotting and aubio (if available) for pitch detection.
    - This is mostly for monophonic or dominant-note detection; full polyphonic piano
      transcription is much more complex and requires ML-based approaches.
    """

    def __init__(self, fs, window_size=4096, hop_size=1024, top_n=10, max_freq=8000.0):
        if not pyqt_available:
            raise RuntimeError('pyqtgraph (and PyQt5/PySide2) is required for live transcription')

        self.fs = fs
        self.window_size = window_size
        self.hop_size = hop_size
        self.top_n = top_n
        self.max_freq = max_freq

        self.buffer = np.zeros(window_size, dtype='float32')
        self.window_func = np.hanning(window_size)
        self.last_hop = np.zeros(hop_size, dtype='float32')

        # aubio pitch detector (optional)
        if aubio_available:
            self.pitch_o = aubio.pitch('yinfft', window_size, hop_size, fs)
            self.pitch_o.set_unit('Hz')
            self.pitch_o.set_silence(-40)
        else:
            self.pitch_o = None

        # Setup pyqtgraph window
        self.app = pg.mkQApp('Live Transcription')
        self.win = pg.GraphicsLayoutWidget(show=True, title='Live Piano Transcription')
        self.win.resize(900, 600)

        # Spectrum plot
        self.spec_plot = self.win.addPlot(row=0, col=0, title='Amplitude spectrum')
        freqs = np.fft.rfftfreq(self.window_size, 1.0 / fs)
        if max_freq is not None:
            mask = freqs <= max_freq
            self.freqs_plot = freqs[mask]
            self.spec_mask = mask
        else:
            self.freqs_plot = freqs
            self.spec_mask = slice(None)
        self.spec_curve = self.spec_plot.plot(self.freqs_plot, np.zeros_like(self.freqs_plot), pen='c')
        self.spec_peaks_scatter = pg.ScatterPlotItem(size=8, brush='r')
        self.spec_plot.addItem(self.spec_peaks_scatter)

        # Time-domain plot
        self.time_plot = self.win.addPlot(row=1, col=0, title='Time domain (window)')
        t = np.arange(self.window_size) / fs
        self.time_curve = self.time_plot.plot(t, np.zeros_like(t), pen='w')
        self.recon_curve = self.time_plot.plot(t, np.zeros_like(t), pen='y')

        # Note display
        self.note_text = pg.TextItem(html='<div style="font-size:20pt">--</div>', anchor=(0, 0))
        self.note_text.setPos(0, 1)
        self.win.addItem(self.note_text, row=0, col=0)

        # Timer to update plots at GUI-friendly rate
        self.timer = QtCore.QTimer()
        interval_ms = max(20, int(1000.0 * self.hop_size / self.fs))
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self.update)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        data = indata[:, 0]
        # update circular buffer
        self.buffer = np.roll(self.buffer, -frames)
        self.buffer[-frames:] = data
        # keep last hop for pitch detection
        if frames >= self.hop_size:
            self.last_hop = data[-self.hop_size:]
        else:
            # combine last part with previous
            self.last_hop = np.roll(self.last_hop, -frames)
            self.last_hop[-frames:] = data

    def update(self):
        # FFT
        windowed = self.buffer * self.window_func
        X = np.fft.rfft(windowed)
        mags = np.abs(X) / len(windowed)
        if len(mags) > 2:
            mags[1:-1] *= 2

        mags_plot = mags[self.spec_mask]
        self.spec_curve.setData(self.freqs_plot, mags_plot)

        # find peaks for simple polyphonic hints (not robust transcription)
        try:
            peaks, _ = find_peaks(mags_plot, height=np.max(mags_plot) * 0.1, distance=3)
            if len(peaks):
                peak_freqs = self.freqs_plot[peaks]
                peak_mags = mags_plot[peaks]
                spots = [{'pos': (f, m)} for f, m in zip(peak_freqs, peak_mags)]
                self.spec_peaks_scatter.setData(spots)
            else:
                self.spec_peaks_scatter.setData([])
        except Exception:
            self.spec_peaks_scatter.setData([])

        # simple reconstruction of top components
        idx = np.argsort(mags)[-self.top_n:][::-1]
        top_freqs = np.fft.rfftfreq(self.window_size, 1.0 / self.fs)[idx]
        top_mags = mags[idx]
        top_phases = np.angle(X[idx])
        t = np.arange(self.window_size) / self.fs
        recon = np.zeros_like(t)
        for A, f, phi in zip(top_mags, top_freqs, top_phases):
            recon += A * np.cos(2 * np.pi * f * t + phi)
        self.time_curve.setData(t, self.buffer)
        self.recon_curve.setData(t, recon)

        # pitch detection (dominant) using aubio if present
        note_str = '--'
        if self.pitch_o is not None:
            # aubio expects float32 array
            buf = aubio.float_type(self.last_hop.tolist())
            pitch = self.pitch_o(buf)[0]
            confidence = self.pitch_o.get_confidence()
            if pitch > 0 and confidence > 0.6:
                note = freq_to_note_name(pitch)
                note_str = f"{note} ({pitch:.1f} Hz, conf {confidence:.2f})"
        else:
            # fallback: use strongest peak
            strongest_idx = np.argmax(mags)
            strongest_freq = np.fft.rfftfreq(self.window_size, 1.0 / self.fs)[strongest_idx]
            if strongest_freq > 0:
                note = freq_to_note_name(strongest_freq)
                note_str = f"{note} ({strongest_freq:.1f} Hz)"

        self.note_text.setText(f"<div style=\"font-size:18pt\">{note_str}</div>")

    def run(self, duration=None):
        try:
            with sd.InputStream(callback=self.audio_callback, channels=1, samplerate=self.fs, blocksize=self.hop_size):
                self.timer.start()
                if duration is None:
                    print(f"Starting live transcription (press Ctrl-C to stop). Window {self.window_size}, hop {self.hop_size}")
                    QtGui.QApplication.instance().exec_()
                else:
                    print(f"Starting live transcription for {duration} seconds")
                    QtGui.QApplication.instance().processEvents()
                    QtCore.QTimer.singleShot(int(duration * 1000), QtGui.QApplication.instance().quit)
                    QtGui.QApplication.instance().exec_()
        except KeyboardInterrupt:
            print('Stopped by user')



