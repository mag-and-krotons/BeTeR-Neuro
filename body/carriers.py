"""The organs' carriers: the still members on which the body's turns ride.

Author of the programme: Abhijit Singh, 19 November 2026.

A carrier is an organ's own rhythm: the heartbeat, the breath, the stomach's slow wave. Each is given here as
  * its phase, phi(t), advancing 2 pi per cycle (the reading of where in its cycle the organ is), and
  * its mechanical drive to the endings of the vagal afferents, d(t) in [0, 1] (wall stretch, lung volume, wall
    tension), from which the afferents' input rate is formed.

Sources, in order of preference:
  heart, breath  observed: one body, one clock (NeuroKit2 'bio_resting_8min_100hz': ECG and respiration belt, 100 Hz,
                 8 min at rest, github.com/neuropsychology/NeuroKit/data). When the MIT-BIH Polysomnographic records
                 are reachable they replace it (ECG, arterial pressure, respiration and EEG of one body).
  stomach        constructed until an EGG record is reachable: a slow wave of 3 cycles per minute whose period
                 wanders (each cycle's period drawn around 20 s with the spread of the heart's own beat-to-beat
                 periods, scaled). Marked 'constructed' in every output that uses it.

Phase from events (heart): phi rises linearly by 2 pi between successive R peaks (the R peak is phase 0).
Phase from a waveform (breath, stomach): the argument of the analytic signal of the band-limited waveform
(Hilbert), unwrapped.

Drives (what the endings feel), each a reading carried as stated:
  baroreceptor  arterial wall stretch after each beat: zero until the pressure wave arrives (R + pre-ejection
                period 0.08 s, a timing read in human hearts), then t/tr exp(1 - t/tr) with rise tr = 0.06 s.
                Shape constructed; timing read. Replaced by the recorded arterial pressure when available.
  stretch       lung volume: the respiration belt, scaled to [0, 1] between its 2nd and 98th percentiles.
  tension       the stomach wall: the slow wave's depolarised half, (1 + cos phi)/2.
"""
from __future__ import annotations

import os

import numpy as np
import scipy.signal as ss

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
NK_FILE = os.path.join(DATA, "raw", "bio_resting_8min_100hz.csv")

PEP_S = 0.08          # R peak to the pressure wave at the aortic arch (pre-ejection period, read in humans)
RISE_S = 0.06


def r_peaks(ecg, fs):
    sos = ss.butter(3, [5, 30], btype="band", fs=fs, output="sos")
    e = ss.sosfiltfilt(sos, ecg)
    pk, _ = ss.find_peaks(e, distance=int(0.4 * fs), height=np.percentile(e, 99) * 0.4)
    return pk / fs


def phase_from_events(events_s, t):
    """2 pi per interval between events, linear inside; NaN outside the first and last event."""
    ph = np.full(t.shape, np.nan)
    k = np.searchsorted(events_s, t, side="right") - 1
    ok = (k >= 0) & (k < len(events_s) - 1)
    a, b = events_s[k[ok]], events_s[k[ok] + 1]
    ph[ok] = 2 * np.pi * (k[ok] + (t[ok] - a) / (b - a))
    return ph


def phase_from_wave(x, fs, band):
    sos = ss.butter(2, band, btype="band", fs=fs, output="sos")
    y = ss.sosfiltfilt(sos, x - np.mean(x))
    return np.unwrap(np.angle(ss.hilbert(y))), y


def scale01(x, lo=2, hi=98):
    a, b = np.percentile(x, [lo, hi])
    return np.clip((x - a) / (b - a), 0, 1)


def baro_drive(beats_s, t):
    d = np.zeros_like(t)
    for tb in beats_s:
        u = t - tb - PEP_S
        m = (u > 0) & (u < 1.5)
        d[m] += (u[m] / RISE_S) * np.exp(1 - u[m] / RISE_S)
    return np.clip(d, 0, 1)


def gastric_constructed(t, rng, mean_period=20.0, rel_sd=0.078):
    """A slow wave whose cycle periods wander with relative spread rel_sd (the heart's RR spread here, 0.083 s on
    1.06 s; used only to give the constructed carrier the variability of a living rhythm)."""
    periods = []
    total = 0.0
    while total < t[-1] + 2 * mean_period:
        p = mean_period * (1 + rel_sd * rng.standard_normal())
        periods.append(p)
        total += p
    starts = np.concatenate([[0.0], np.cumsum(periods)]) - rng.uniform(0, mean_period)
    return phase_from_events(starts, t)


def load(duration_s=300.0, fs=1000.0, start_s=None, seed=7):
    """Carriers on a grid of 1/fs, starting at the photosensor onset of the NeuroKit record (its 8-min window)."""
    import pandas as pd
    b = pd.read_csv(NK_FILE)
    fs0 = 100.0
    ph = b["PhotoSensor"].values
    onset = np.nonzero(np.abs(np.diff(ph)) > 1)[0]
    t0 = (onset[0] + 1) / fs0 if start_s is None else start_s
    beats = r_peaks(b["ECG"].values, fs0) - t0
    rsp = b["RSP"].values
    t_rsp = np.arange(len(rsp)) / fs0 - t0
    t = np.arange(0, duration_s, 1 / fs)
    heart_phase = phase_from_events(beats, t)
    rph, rwave = phase_from_wave(rsp, fs0, (0.05, 0.6))
    breath_phase = np.interp(t, t_rsp, rph)
    lung = np.interp(t, t_rsp, scale01(rsp))
    rng = np.random.default_rng(seed)
    gastric_phase = gastric_constructed(t, rng)
    return {
        "t": t, "fs": fs,
        "heart": {"phase": heart_phase, "drive": baro_drive(beats, t), "events_s": beats[(beats >= 0) &
                  (beats < duration_s)], "source": "observed ECG (NeuroKit2 bio_resting_8min_100hz)"},
        "breath": {"phase": breath_phase, "drive": lung, "source": "observed respiration belt (same record)"},
        "stomach": {"phase": gastric_phase, "drive": (1 + np.cos(gastric_phase)) / 2,
                    "source": "constructed slow wave, 3/min (no EGG reachable)"},
    }


def frequencies(car) -> dict:
    """Mean frequency of each carrier from its phase: total phase advanced / 2 pi / time."""
    out = {}
    t = car["t"]
    for k in ("heart", "breath", "stomach"):
        p = car[k]["phase"]
        ok = np.isfinite(p)
        out[k] = float((p[ok][-1] - p[ok][0]) / (2 * np.pi) / (t[ok][-1] - t[ok][0]))
    return out
