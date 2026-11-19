"""The same reading, on a recorded body: EEG read against its own heart and breath (MIT-BIH Polysomnographic
Database, PhysioNet slpdb: ECG, invasive arterial pressure, one EEG lead and respiration, one body, one clock).

Author of the programme: Abhijit Singh, 19 November 2026.

For each record:
  heart phase   from R peaks of the recorded ECG (0 at each R peak, linear between)
  breath phase  analytic-signal phase of the recorded respiration (0.05-0.6 Hz)
  EEG           band 0.5-30 Hz. Read against each carrier exactly as the model's populations are
                (reading.read_signal): complex locking Z = 2 <x exp(-i phi)> / sd(x), the in-phase correlation
                r0 = corr(x, cos phi), the quadrature r90 = corr(x, sin phi), and time-shifted controls.
  pressure      the recorded arterial pressure read against the heart phase: the pulse's own arrival, a phase
                reading that needs no shape assumed (it replaces the model's constructed pulse when present).
  ECG itself    read against the heart phase: what an instantaneous reading of the EEG would share with the
                heart's own electric field (the field reaches the scalp at zero delay).

Run:  python3 recorded.py fetch    (downloads slpdb into data/slpdb; needs physionet.org reachable)
      python3 recorded.py read     (reads every record that is present)
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import scipy.signal as ss

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "body"))
from carriers import phase_from_events, phase_from_wave, r_peaks  # noqa: E402

DB = os.path.join(ROOT, "data", "slpdb")


def fetch():
    import wfdb
    os.makedirs(DB, exist_ok=True)
    wfdb.dl_database("slpdb", DB)


def read_signal(x, t, phases, rng, n_shift=200, min_shift=20.0, good=None):
    """x on grid t. The complete reading against each carrier, with circular time-shift controls.
    Samples where good is False are left out (set to the mean after standardising)."""
    good = np.ones(len(x), bool) if good is None else good
    x = x.astype(float).copy()
    x[~good] = np.nan
    x = (x - np.nanmean(x)) / np.nanstd(x)
    x[~good] = 0.0
    out = {}
    span = t[-1] - t[0]
    fs = 1 / (t[1] - t[0])
    shifts = (rng.uniform(min_shift, span - min_shift, n_shift) * fs).astype(int)
    for k, ph in phases.items():
        ok = np.isfinite(ph) & good
        e = np.where(ok, np.exp(-1j * np.where(ok, ph, 0.0)), 0.0)
        c, s = e.real, -e.imag
        n = ok.sum()

        def read(y):
            z = 2 * np.sum(y * e) / n
            yy = y[ok]
            a = np.dot(yy, c[ok]) / math.sqrt(np.dot(yy, yy) * np.dot(c[ok], c[ok]))
            b = np.dot(yy, s[ok]) / math.sqrt(np.dot(yy, yy) * np.dot(s[ok], s[ok]))
            return z, a, b
        z, r0, r90 = read(x)
        nz, n0, npair = [], [], []
        for sh in shifts:
            zz, a, b = read(np.roll(x, sh))
            nz.append(abs(zz))
            n0.append(abs(a))
            npair.append(math.hypot(a, b))
        nz, n0, npair = map(np.array, (nz, n0, npair))
        out[k] = {"Z_abs": float(abs(z)), "Z_phase_deg": float(np.degrees(np.angle(np.conj(z)))),
                  "p_Z": float(np.mean(nz >= abs(z))), "Z_abs_null_95": float(np.percentile(nz, 95)),
                  "r_in_phase": float(r0), "r_quadrature": float(r90), "r_pair": float(math.hypot(r0, r90)),
                  "p_r_in_phase": float(np.mean(n0 >= abs(r0))), "p_r_pair": float(np.mean(npair >= math.hypot(r0, r90))),
                  "in_phase_share_of_pair": float(r0 ** 2 / (r0 ** 2 + r90 ** 2)) if (r0 or r90) else None}
    return out


def locked_average(x, beats_idx, fs, good, pre=0.2, post=1.0):
    """The heart-locked average of x (each beat from -pre to +post s), beats whose window is all good."""
    a, b = int(pre * fs), int(post * fs)
    idx = [i for i in beats_idx if i - a >= 0 and i + b < len(x) and good[i - a:i + b].all()]
    ep = np.stack([x[i - a:i + b] for i in idx])
    return np.arange(-a, b) / fs, ep.mean(0), ep.std(0) / math.sqrt(len(idx)), len(idx)


def bad_mask(x, fs, z=8.0, pad_s=1.0):
    """Saturated or broken stretches: |x - median| > z robust SDs, flat runs of 2 s, NaN; padded by pad_s."""
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826 + 1e-12
    bad = ~np.isfinite(x) | (np.abs(x - med) > z * mad)
    d = np.abs(np.diff(x, prepend=x[0])) < 1e-12
    run = np.convolve(d.astype(float), np.ones(int(2 * fs)), "same") >= 2 * fs - 1
    bad |= run
    k = int(pad_s * fs)
    if k:
        bad = np.convolve(bad.astype(float), np.ones(2 * k + 1), "same") > 0
    return bad


def read_record(name, rng, n_shift=200):
    import wfdb
    rec = wfdb.rdrecord(os.path.join(DB, name))
    fs = rec.fs
    sig = {n: rec.p_signal[:, i] for i, n in enumerate(rec.sig_name)}
    n = rec.sig_len
    t = np.arange(n) / fs
    ecg, bp = sig["ECG"], sig["BP"]
    eeg_key = next(k for k in sig if k.startswith("EEG"))
    eeg = sig[eeg_key]
    resp_key = next(k for k in sig if k.startswith("Resp"))
    # the heart's events: the database's own beat annotations (normal beats), checked against our detector
    ann = wfdb.rdann(os.path.join(DB, name), "ecg")
    beats_idx = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym == "N"])
    beats = beats_idx / fs
    ours = r_peaks(np.nan_to_num(ecg), fs)
    j = np.searchsorted(ours, beats)
    dmin = np.minimum(np.abs(beats - ours[np.clip(j - 1, 0, len(ours) - 1)]),
                      np.abs(beats - ours[np.clip(j, 0, len(ours) - 1)]))
    phases = {"heart": phase_from_events(beats, t)}
    rr = np.diff(beats)
    phases["heart"][np.isfinite(phases["heart"])] *= 1.0
    # a gap of more than 2 s between annotated beats (artifact, ectopic run) is not read
    k_ev = np.searchsorted(beats, t, side="right") - 1
    okk = (k_ev >= 0) & (k_ev < len(rr))
    long_gap = np.zeros(n, bool)
    long_gap[okk] = rr[k_ev[okk]] > 2.0
    phases["heart"][long_gap] = np.nan
    rph, _ = phase_from_wave(np.nan_to_num(sig[resp_key]), fs, (0.05, 0.6))
    phases["breath"] = rph
    sos = ss.butter(2, [0.5, 30], btype="band", fs=fs, output="sos")
    bad_eeg = bad_mask(eeg, fs)
    bad_bp = bad_mask(bp, fs, z=10.0)
    eeg_f = ss.sosfiltfilt(sos, np.where(bad_eeg, np.nanmedian(eeg), eeg))
    ecg_f = ss.sosfiltfilt(sos, np.nan_to_num(ecg))
    good = ~bad_eeg
    out = {"record": name, "fs": fs, "hours": n / fs / 3600, "eeg_lead": eeg_key, "resp": resp_key,
           "beats_normal": int(len(beats)), "heart_Hz": float(1 / np.median(rr)),
           "breath_Hz": float((rph[-1] - rph[0]) / (2 * np.pi) / (t[-1])),
           "our_R_within_20ms_of_annotation": float(np.mean(dmin <= 0.02)),
           "eeg_good_fraction": float(good.mean()), "bp_good_fraction": float((~bad_bp).mean())}
    # readings (decimated to 50 Hz after the 30 Hz band: the carriers are below 2 Hz)
    dec = 5
    tt = t[::dec]
    ph_d = {k: v[::dec] for k, v in phases.items()}
    out["EEG"] = read_signal(eeg_f[::dec], tt, ph_d, rng, n_shift, good=good[::dec])
    out["ECG"] = read_signal(ecg_f[::dec], tt, ph_d, rng, n_shift)
    # the heart's own electric field reaches the scalp instantly, shaped as the ECG: its zero-delay share of the EEG,
    # fitted over the whole record, taken out; what remains is read the same way
    g = good & np.isfinite(ecg_f)
    a_glob = float(np.dot(eeg_f[g], ecg_f[g]) / np.dot(ecg_f[g], ecg_f[g]))
    eeg_r = eeg_f - a_glob * ecg_f
    out["field_scale_whole_record"] = a_glob
    out["EEG_minus_field"] = read_signal(eeg_r[::dec], tt, ph_d, rng, n_shift, good=good[::dec])
    out["BP"] = read_signal(np.where(bad_bp, np.nanmedian(bp), bp)[::dec], tt, ph_d, rng, n_shift, good=~bad_bp[::dec])
    # heart-locked averages at 250 Hz
    lt, eeg_avg, eeg_se, n_e = locked_average(eeg_f, beats_idx, fs, good)
    _, ecg_avg, _, _ = locked_average(ecg_f, beats_idx, fs, good)
    _, bp_avg, _, n_b = locked_average(np.where(bad_bp, np.nanmedian(bp), bp), beats_idx, fs, ~bad_bp)
    # the heart's own field reaches the scalp at zero delay with the ECG's shape: scale the ECG average to the EEG
    # average over the QRS (-40 to +60 ms) and take it out; what remains is what arrives later
    q = (lt >= -0.04) & (lt <= 0.06)
    a = float(np.dot(eeg_avg[q], ecg_avg[q]) / np.dot(ecg_avg[q], ecg_avg[q]))
    resid = eeg_avg - a * ecg_avg
    base = (lt >= -0.2) & (lt < -0.05)
    resid = resid - resid[base].mean()
    noise = float(np.mean(eeg_se[lt > 0]))
    w = (lt > 0.1) & (lt < 0.8)
    i_pk = np.argmax(np.abs(resid) * w)
    bp_peak_t = float(lt[np.argmax(np.where(lt > 0, bp_avg, -np.inf))])
    bp_d = np.gradient(bp_avg, lt)
    bp_upstroke_t = float(lt[np.argmax(np.where((lt > 0) & (lt < 0.5), bp_d, -np.inf))])
    se_r = noise
    ext = []
    for i in range(1, len(lt) - 1):
        if 0.05 <= lt[i] <= 0.8 and abs(resid[i]) >= abs(resid[i - 1]) and abs(resid[i]) >= abs(resid[i + 1]) \
                and abs(resid[i]) > 4 * se_r:
            ext.append((float(lt[i]), float(resid[i] * 1e3), float(abs(resid[i]) / se_r)))
    out["heart_locked"] = {"residual_extrema_s_uV_se": ext,
        "beats_used_eeg": n_e, "beats_used_bp": n_b, "field_scale_ecg_to_eeg": a,
        "residual_peak_s": float(lt[i_pk]), "residual_peak_uV": float(resid[i_pk] * 1e3),
        "residual_peak_over_se": float(abs(resid[i_pk]) / noise),
        "pressure_upstroke_s": bp_upstroke_t, "pressure_peak_s": bp_peak_t,
        "t_s": lt[::5].tolist(), "eeg_uV": (eeg_avg[::5] * 1e3).tolist(), "residual_uV": (resid[::5] * 1e3).tolist(),
        "ecg_scaled_uV": (a * ecg_avg[::5] * 1e3).tolist(), "bp_mmHg": bp_avg[::5].tolist(),
        "se_uV": (eeg_se[::5] * 1e3).tolist()}
    return out


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "fetch":
        fetch()
        sys.exit(0)
    rng = np.random.default_rng(9)
    recs = open(os.path.join(DB, "RECORDS")).read().split()
    res = []
    for r in recs:
        res.append(read_record(r, rng))
        e = res[-1]["EEG"]["heart"]
        print(r, f"heart {res[-1]['heart_Hz']:.2f} Hz | EEG |Z| {e['Z_abs']:.3f} at {e['Z_phase_deg']:.0f} deg "
              f"p {e['p_Z']:.3f} r0 {e['r_in_phase']:+.3f} r90 {e['r_quadrature']:+.3f} | residual peak "
              f"{res[-1]['heart_locked']['residual_peak_s']:.3f} s ({res[-1]['heart_locked']['residual_peak_over_se']:.1f} se)",
              flush=True)
    with open(os.path.join(ROOT, "results", "recorded_slpdb_reading.json"), "w") as f:
        json.dump(res, f, indent=1)

