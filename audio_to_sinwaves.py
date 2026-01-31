#!/usr/bin/env python3
"""
Record from the default microphone, compute FFT, identify top-N frequency components,
reconstruct the signal as a sum of sinusoids, and plot the amplitude spectrum and
original vs reconstructed time-domain signals.

Usage:
    python audio_to_sinwaves.py --duration 3 --top 10

Dependencies:
    pip install -r requirements.txt
"""

import argparse
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def record_audio(duration=3.0, fs=44100):
    print(f"Recording {duration:.2f} s at {fs} Hz...")
    data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return data.flatten(), fs


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


def plot_results(x, fs, result, max_freq_plot=None):
    freqs = result['freqs']
    mags = result['mags']
    top_freqs = result['top_freqs']
    top_mags = result['top_mags']
    recon = result['recon']
    t = result['t']

    plt.figure(figsize=(12, 6))

    # Spectrum
    plt.subplot(2, 1, 1)
    if max_freq_plot is None:
        plt.plot(freqs, mags, color='C0')
    else:
        mask = freqs <= max_freq_plot
        plt.plot(freqs[mask], mags[mask], color='C0')
    plt.scatter(top_freqs, top_mags, color='C1', zorder=3, label='Top components')
    for f, A in zip(top_freqs, top_mags):
        plt.text(f, A, f"{f:.1f} Hz", fontsize=8, ha='center', va='bottom')
    plt.title('Amplitude spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # Time domain: show original vs reconstruction (plot first 50 ms for clarity)
    plt.subplot(2, 1, 2)
    ms50 = int(min(len(t), int(0.05 * fs)))
    plt.plot(t[:ms50], x[:ms50], label='Original', alpha=0.7)
    plt.plot(t[:ms50], recon[:ms50], label='Reconstructed (top components)', alpha=0.7)
    plt.title('Time domain (first 50 ms)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Record microphone, decompose to sinusoids, and plot amplitudes')
    parser.add_argument('--duration', type=float, default=3.0, help='Recording duration in seconds')
    parser.add_argument('--fs', type=int, default=44100, help='Sampling rate (Hz)')
    parser.add_argument('--top', type=int, default=10, help='Number of top sinusoids to keep for reconstruction')
    parser.add_argument('--maxfreq', type=float, default=8000.0, help='Max frequency to plot (Hz)')
    args = parser.parse_args()

    x, fs = record_audio(duration=args.duration, fs=args.fs)
    result = analyze_and_reconstruct(x, fs, top_n=args.top, max_freq=args.maxfreq)

    print('\nTop components:')
    for f, A in zip(result['top_freqs'], result['top_mags']):
        print(f"  {f:.1f} Hz — amplitude {A:.5f}")

    plot_results(x, fs, result, max_freq_plot=args.maxfreq)


if __name__ == '__main__':
    main()
