"""The reading at the horizon: every population read against every carrier, both members kept.

Author of the programme: Abhijit Singh, 19 November 2026.

The same functions read the model's output and recorded signals.

For a spike train (or any signal) and a carrier with phase phi(t):

  complete reading   Z = <exp(i phi(t_s))> over the spikes: |Z| how strongly the turns ride the carrier, arg Z where
                     in the carrier's cycle they ride. I = Re Z (in phase with the carrier's zero), Q = Im Z
                     (quadrature). Nothing is discarded: (I, Q) is the whole first harmonic of the locking.
  in-phase reading   r0 = zero-lag correlation of the population rate with cos(phi): what an instantaneous
                     (in-phase) connectivity reading reports. Its partner in quadrature, r90 = correlation with
                     sin(phi), is computed beside it; sqrt(r0^2 + r90^2) is what the pair holds.
  control            every quantity recomputed with the spike train shifted in time by a random offset (>= 20 s,
                     circularly), 200 times: the size a reading reaches when the turns are not tied to the carrier.
                     A reading is reported with the fraction of shifted trains that reach it (p).

Signals are binned at 10 ms for correlations. Phases of the carriers come on a 100 Hz grid.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

CARRIERS = ("heart", "breath", "stomach")


def phase_at(t_grid, phase_grid, t):
    ok = np.isfinite(phase_grid)
    return np.interp(t, t_grid[ok], phase_grid[ok], left=np.nan, right=np.nan)


def locking(spike_t, t_grid, phase_grid):
    ph = phase_at(t_grid, phase_grid, spike_t)
    ph = ph[np.isfinite(ph)]
    if len(ph) == 0:
        return 0j, 0
    return np.mean(np.exp(1j * ph)), len(ph)


def binned_rate(spike_t, n_neurons, t_end, bin_s=0.01):
    edges = np.arange(0, t_end + bin_s / 2, bin_s)
    c, _ = np.histogram(spike_t, edges)
    return c / (n_neurons * bin_s), edges[:-1] + bin_s / 2


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / den) if den > 0 else 0.0


def read_population(spike_t, n_neurons, t_grid, phases, t_end, rng, n_shift=200, min_shift=20.0, settle=5.0,
                    bin_s=0.005):
    """All carriers for one population. spike_t in s. Spikes before `settle` s are left out (start-up).

    Spikes are counted in 5 ms bins; the carrier phase is taken at each bin's centre (at the heart's 0.96 Hz a bin
    spans 1.7 degrees, which shortens |Z| by a factor sinc(0.85 deg) = 0.99996). A time shift of the train is a
    circular shift of the counts by whole bins."""
    s = np.sort(spike_t[(spike_t >= settle) & (spike_t < t_end)])
    out = {"spikes": int(len(s)), "rate_Hz": float(len(s) / (n_neurons * (t_end - settle)))}
    edges = np.arange(settle, t_end + bin_s / 2, bin_s)
    counts, _ = np.histogram(s, edges)
    tc = edges[:-1] + bin_s / 2
    k10 = int(round(0.01 / bin_s))                       # 10 ms bins for the correlations
    nb = (len(counts) // k10) * k10
    span_bins = len(counts)
    shifts = (rng.uniform(min_shift, (t_end - settle) - min_shift, n_shift) / bin_s).astype(int)
    for k, ph_grid in phases.items():
        ph = phase_at(t_grid, ph_grid, tc)
        ok = np.isfinite(ph)
        e = np.where(ok, np.exp(1j * np.where(ok, ph, 0.0)), 0.0)
        z = np.sum(counts * e) / max(np.sum(counts[ok]), 1)
        r10 = counts[:nb].reshape(-1, k10).sum(1).astype(float)
        ph10 = ph[:nb].reshape(-1, k10)[:, k10 // 2]
        ok10 = np.isfinite(ph10)
        c10, s10 = np.cos(ph10[ok10]), np.sin(ph10[ok10])
        r0 = corr(r10[ok10], c10)
        r90 = corr(r10[ok10], s10)
        null_z, null_r0, null_pair = [], [], []
        for sh in shifts:
            cs = np.roll(counts, sh)
            null_z.append(abs(np.sum(cs * e) / max(np.sum(cs[ok]), 1)))
            rr = cs[:nb].reshape(-1, k10).sum(1).astype(float)[ok10]
            a, b = corr(rr, c10), corr(rr, s10)
            null_r0.append(abs(a))
            null_pair.append(math.hypot(a, b))
        null_z, null_r0, null_pair = map(np.array, (null_z, null_r0, null_pair))
        out[k] = {
            "Z_abs": float(abs(z)), "Z_phase_deg": float(np.degrees(np.angle(z))),
            "I": float(z.real), "Q": float(z.imag),
            "p_Z": float(np.mean(null_z >= abs(z))), "Z_abs_null_95": float(np.percentile(null_z, 95)),
            "r_in_phase": r0, "r_quadrature": r90, "r_pair": float(math.hypot(r0, r90)),
            "p_r_in_phase": float(np.mean(null_r0 >= abs(r0))),
            "p_r_pair": float(np.mean(null_pair >= math.hypot(r0, r90))),
            "r_pair_null_95": float(np.percentile(null_pair, 95)),
            "r_in_phase_null_95": float(np.percentile(null_r0, 95)),
        }
    return out


def profile_peaks(spike_t, t_grid, phase_grid, f_hz, nbins=36, settle=5.0):
    """The whole profile of the turns over the carrier's cycle, and its peaks (in degrees and in seconds after the
    carrier's zero). One number, arg Z, is the average of every arrival; the profile keeps each one."""
    s = spike_t[spike_t >= settle]
    p = np.mod(phase_at(t_grid, phase_grid, s), 2 * np.pi)
    p = p[np.isfinite(p)]
    h_, edges = np.histogram(p, np.linspace(0, 2 * np.pi, nbins + 1))
    prof = h_ / h_.mean()
    mids = np.degrees(edges[:-1] + np.diff(edges) / 2)
    ext = np.concatenate([prof[-1:], prof, prof[:1]])
    pk = [i for i in range(nbins) if ext[i + 1] > ext[i] and ext[i + 1] >= ext[i + 2] and prof[i] > 1.2]
    return {"profile": [float(x) for x in prof], "phase_deg": [float(x) for x in mids],
            "peaks_deg": [float(mids[i]) for i in pk], "peaks_height": [float(prof[i]) for i in pk],
            "peaks_s_after_zero": [float(mids[i] / 360.0 / f_hz) for i in pk]}


def flight_phase(f_hz, delay_s):
    """The phase a pure delay turns a carrier of frequency f by, wrapped to (-180, 180]."""
    d = (360.0 * f_hz * delay_s) % 360.0
    return d - 360.0 if d > 180 else d


def front_reading(spike_t, n_neurons, t_end, levels, hold_s, settle=5.0, lags=np.arange(0.0, 0.31, 0.01)):
    """How the sensory stimulus (held levels) appears in a population's rate: correlation at each lag."""
    r, tc = binned_rate(np.sort(spike_t), n_neurons, t_end)
    stim = levels[np.minimum((tc / hold_s).astype(int), len(levels) - 1)]
    keep = tc >= settle
    out = []
    for lag in lags:
        shift = int(round(lag / 0.01))
        a = r[keep][shift:]
        b = stim[keep][:len(a)]
        out.append(corr(a, b))
    out = np.array(out)
    return {"r_at_lag": [float(x) for x in out], "lags_s": [float(x) for x in lags],
            "best_lag_s": float(lags[int(np.argmax(out))]), "best_r": float(out.max())}


def members(npz, group, t_settle=5.0):
    """The excitatory and inhibitory members of the input to a recorded group, as currents (pA, inward negative):
    I_E = g_ex (V - E_ex), I_I = g_in (V - E_in); mean over recorded neurons and time, and how much cancels."""
    t = npz[f"mm_{group}_times"] * 1e-3
    snd = npz[f"mm_{group}_senders"]
    v = npz[f"mm_{group}_V_m"].astype(float)
    ge = npz[f"mm_{group}_g_ex"].astype(float)
    gi = npz[f"mm_{group}_g_in"].astype(float)
    keep = (t >= t_settle) & (v < -40)                       # outside spikes
    ie = ge[keep] * (v[keep] - 0.0)
    ii = gi[keep] * (v[keep] + 80.0)
    net = ie + ii
    return {"I_E_mean_pA": float(ie.mean()), "I_I_mean_pA": float(ii.mean()), "net_mean_pA": float(net.mean()),
            "cancelled_fraction_of_E": float(1 - abs(net.mean()) / abs(ie.mean())),
            "net_sd_pA": float(net.std()), "V_mean_mV": float(v[keep].mean()),
            "neurons": int(len(np.unique(snd)))}


def read_run(stem, root, n_shift=200, seed=5):
    npz = np.load(os.path.join(root, "results", stem + ".npz"))
    meta = json.load(open(os.path.join(root, "results", stem + ".json")))
    t_end = meta["duration_s"]
    t_grid = npz["car_t"]
    phases = {k: npz[f"car_{k}_phase"] for k in CARRIERS}
    freqs = meta["carrier_frequencies_Hz"]
    rng = np.random.default_rng(seed)
    gids = meta["gids"]
    res = {"carrier_frequencies_Hz": freqs, "carrier_sources": meta["carrier_sources"], "populations": {}}
    groups = {name: (npz[f"spk_{name}_t"] * 1e-3, g[1] - g[0] + 1) for name, g in gids.items()}
    # sub-areas of the cortex
    lo = gids["ctx_E"][0]
    t_ctx, id_ctx = npz["spk_ctx_E_t"] * 1e-3, npz["spk_ctx_E_id"] - lo
    for sub, (a, b) in {"ctx_insula": (0, 400), "ctx_sensory": (400, 800), "ctx_assoc": (800, 1600)}.items():
        m = (id_ctx >= a) & (id_ctx < b)
        groups[sub] = (t_ctx[m], b - a)
    for name, (st, n) in groups.items():
        res["populations"][name] = read_population(st, n, t_grid, phases, t_end, rng, n_shift=n_shift)
        res["populations"][name]["front"] = front_reading(st, n, t_end, npz["front_levels"],
                                                          meta["cfg"]["front_hold_s"])
        res["populations"][name]["heart_profile"] = profile_peaks(st, t_grid, phases["heart"], freqs["heart"])
    res["members"] = {}
    for g in ("insula", "sensory", "assoc", "ctx_I", "nts_cardio_E"):
        if f"mm_{g}_times" in npz.files:
            res["members"][g] = members(npz, g)
    res["flight_phase_deg"] = {
        "heart_via_C_250ms_plus_PEP": flight_phase(freqs["heart"], 0.25 + 0.08),
        "heart_via_A_12.5ms_plus_PEP": flight_phase(freqs["heart"], 0.0125 + 0.08),
        "breath_via_A_10ms": flight_phase(freqs["breath"], 0.010),
        "stomach_via_C_500ms": flight_phase(freqs["stomach"], 0.5)}
    return res


if __name__ == "__main__":
    import sys
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    stem = sys.argv[1] if len(sys.argv) > 1 else "body_run"
    r = read_run(stem, root)
    with open(os.path.join(root, "results", stem + "_reading.json"), "w") as f:
        json.dump(r, f, indent=1)
    for name, p in r["populations"].items():
        line = f"{name:16s} {p['rate_Hz']:6.2f} Hz |"
        for k in CARRIERS:
            q = p[k]
            line += (f" {k[:5]} |Z| {q['Z_abs']:.3f} @{q['Z_phase_deg']:6.0f} (p {q['p_Z']:.3f})"
                     f" r0 {q['r_in_phase']:+.3f} r90 {q['r_quadrature']:+.3f} |")
        print(line)
    print(json.dumps(r["members"], indent=1))
    print(json.dumps(r["flight_phase_deg"], indent=1))
