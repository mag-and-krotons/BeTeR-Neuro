"""Figures for the record. Each figure is drawn from results/ and from the model itself; nothing is retouched.

Author of the programme: Abhijit Singh, 19 November 2026.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "results", "figures")
sys.path.insert(0, os.path.join(ROOT, "cell"))
sys.path.insert(0, os.path.join(ROOT, "reading"))

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C_NA, C_K, C_V, C_P = "#c0392b", "#2c6fbb", "#222222", "#8e44ad"


def fig_cell():
    from neuron import h
    from cell import Cell, close_rest, ma_cm2_to_pa
    rest = close_rest()
    c = Cell("live", rest)
    st = h.IClamp(c.soma(0.5))
    st.delay, st.dur, st.amp = 5.0, 1.0, 4.0
    c._keep.append(st)
    c.record(0.005)
    out = c.run(20.0, atol=1e-9, rtol=1e-10)
    t = out["t"]
    ina = ma_cm2_to_pa(out["ina_hh"]) + ma_cm2_to_pa(out["ina_l"])
    ik = ma_cm2_to_pa(out["ik_hh"]) + ma_cm2_to_pa(out["ik_l"])
    qna = -np.cumsum(ina) * 0.005 * 1e-3
    qk = np.cumsum(ik) * 0.005 * 1e-3
    ex = json.load(open(os.path.join(ROOT, "results", "cell_exchange.json")))
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.0))
    a = ax[0]
    a.plot(t, out["v"], color=C_V, lw=1)
    a.set_xlabel("ms")
    a.set_ylabel("V (mV)")
    a.set_title("one spike from the closed rest (-68.45 mV)")
    a = ax[1]
    a.plot(t, -ina / 1e3, color=C_NA, lw=1, label="Na in (nA)")
    a.plot(t, ik / 1e3, color=C_K, lw=1, label="K out (nA)")
    a.set_xlim(4, 10)
    a.set_xlabel("ms")
    a.legend(frameon=False)
    a.set_title("the two members")
    a = ax[2]
    a.plot(t, qna, color=C_NA, lw=1.2, label="Na charge in")
    a.plot(t, qk, color=C_K, lw=1.2, label="K charge out")
    a.plot(t, 200 * (out["v"] - out["v"][0]) * 1e-3, color=C_V, lw=1, ls="--", label="C dV (what V reads)")
    a.set_xlabel("ms")
    a.set_ylabel("pC")
    a.legend(frameon=False, loc="upper left")
    sp = ex["spike"]["live"]
    a.set_title(f"V reads {100 * sp['voltage_reads_fraction_of_Na_charge']:.0f}% of the Na charge (rho {sp['rho']:.2f})", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_cell_pair.png"))
    plt.close(fig)


ORDER = ["aff_baroA", "aff_baroC", "aff_stretch", "aff_gastric", "nts_cardio_E", "nts_resp_E", "nts_gastric_E",
         "relay", "ctx_insula", "ctx_sensory", "ctx_assoc", "ctx_I", "eff"]
LABEL = {"aff_baroA": "baro A", "aff_baroC": "baro C", "aff_stretch": "lung", "aff_gastric": "gastric",
         "nts_cardio_E": "NTS cardio", "nts_resp_E": "NTS resp", "nts_gastric_E": "NTS gastric", "relay": "relay",
         "ctx_insula": "insula", "ctx_sensory": "sensory", "ctx_assoc": "assoc", "ctx_I": "ctx I", "eff": "efferent"}


def fig_body(stem="body_run"):
    r = json.load(open(os.path.join(ROOT, "results", stem + "_reading.json")))
    P = r["populations"]
    carriers = ("heart", "breath", "stomach")
    fig = plt.figure(figsize=(11, 7.2))
    for j, k in enumerate(carriers):
        a = fig.add_subplot(2, 3, j + 1, projection="polar")
        for name in ORDER:
            q = P[name][k]
            sig = q["p_Z"] < 0.01
            th = np.radians(q["Z_phase_deg"])
            a.plot([0, th], [0, q["Z_abs"]], color=C_V if sig else "#bbbbbb", lw=1)
            a.plot(th, q["Z_abs"], "o", ms=3, color=C_NA if name.startswith("ctx") else C_V if sig else "#bbbbbb")
            if sig and q["Z_abs"] > 0.04:
                a.text(th, q["Z_abs"] * 1.05, LABEL[name], fontsize=6)
        a.set_title(f"{k}, {r['carrier_frequencies_Hz'][k]:.3f} Hz\nwhere in the cycle the spikes ride",
                    fontsize=8)
        a.set_rlabel_position(135)
        a.tick_params(labelsize=6)
    for j, k in enumerate(carriers):
        a = fig.add_subplot(2, 3, 4 + j)
        x = np.arange(len(ORDER))
        r0 = [abs(P[n][k]["r_in_phase"]) for n in ORDER]
        rp = [P[n][k]["r_pair"] for n in ORDER]
        a.bar(x - 0.2, rp, 0.4, color=C_P, label="both members, sqrt(r0^2 + r90^2)")
        a.bar(x + 0.2, r0, 0.4, color="#e0a800", label="in-phase reading |r0|")
        a.plot(x, [P[n][k]["r_pair_null_95"] for n in ORDER], "_", color="k", ms=8, label="shifted 95%")
        a.set_xticks(x)
        a.set_xticklabels([LABEL[n] for n in ORDER], rotation=70, fontsize=7)
        a.set_title(f"{k}: rate read in phase vs whole", fontsize=8)
        if j == 0:
            a.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_body_reading.png"))
    plt.close(fig)


def fig_profiles(stem="body_run", names=("aff_baroC", "nts_cardio_E", "relay", "ctx_insula", "ctx_sensory", "eff")):
    import reading as R
    npz = np.load(os.path.join(ROOT, "results", stem + ".npz"))
    meta = json.load(open(os.path.join(ROOT, "results", stem + ".json")))
    gids = meta["gids"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.0))
    for j, k in enumerate(("heart", "breath", "stomach")):
        ph = npz[f"car_{k}_phase"]
        tg = npz["car_t"]
        bins = np.linspace(0, 2 * np.pi, 25)
        for name in names:
            if name.startswith("ctx_"):
                lo = gids["ctx_E"][0]
                rng_ = {"ctx_insula": (0, 400), "ctx_sensory": (400, 800)}[name]
                st = npz["spk_ctx_E_t"] * 1e-3
                idd = npz["spk_ctx_E_id"] - lo
                st = st[(idd >= rng_[0]) & (idd < rng_[1])]
            else:
                st = npz[f"spk_{name}_t"] * 1e-3
            st = st[st > 5]
            p = np.mod(R.phase_at(tg, ph, st), 2 * np.pi)
            p = p[np.isfinite(p)]
            hcount, _ = np.histogram(p, bins)
            prof = hcount / hcount.mean()
            ax[j].plot(np.degrees(bins[:-1] + np.diff(bins) / 2), prof, lw=1, label=LABEL[name])
        ax[j].axhline(1, color="#999", lw=0.5)
        ax[j].set_xlabel(f"{k} phase (deg; 0 = " + {"heart": "R peak", "breath": "peak of the breath wave",
                                                     "stomach": "slow-wave peak"}[k] + ")")
        ax[j].set_xlim(0, 360)
        ax[j].set_xticks([0, 90, 180, 270, 360])
        if j == 0:
            ax[j].set_ylabel("spikes per phase bin / mean")
            ax[j].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_phase_profiles.png"))
    plt.close(fig)


def fig_control():
    a = json.load(open(os.path.join(ROOT, "results", "body_run_reading.json")))["populations"]
    b = json.load(open(os.path.join(ROOT, "results", "body_flat_reading.json")))["populations"]
    names = [n for n in ORDER if n in b]
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].bar(x - 0.2, [a[n]["rate_Hz"] for n in names], 0.4, color=C_V, label="carriers as recorded")
    ax[0].bar(x + 0.2, [b[n]["rate_Hz"] for n in names], 0.4, color="#bbbbbb", label="each drive held at its mean")
    ax[0].set_ylabel("rate (Hz)")
    ax[0].set_title("the value: mean rate", fontsize=8)
    ax[0].legend(frameon=False, fontsize=7)
    for j, (k, c) in enumerate((("heart", C_NA), ("breath", C_K), ("stomach", C_P))):
        ax[1].bar(x + (j - 1) * 0.27, [a[n][k]["Z_abs"] for n in names], 0.27, color=c, label=k)
        ax[1].plot(x + (j - 1) * 0.27, [b[n][k]["Z_abs"] for n in names], "_", color="k", ms=6)
    ax[1].set_title("the phase: |Z| per carrier (bars recorded; ticks: drive held at its mean)", fontsize=8)
    ax[1].legend(frameon=False, fontsize=7)
    for a_ in ax:
        a_.set_xticks(x)
        a_.set_xticklabels([LABEL[n] for n in names], rotation=70, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_control_mean_drive.png"))
    plt.close(fig)



def fig_recorded():
    R = json.load(open(os.path.join(ROOT, "results", "recorded_slpdb_reading.json")))
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    for j, r in enumerate(R):
        h = r["heart_locked"]
        t = np.array(h["t_s"])
        res = np.array(h["residual_uV"]) / np.array(h["se_uV"])
        ax[0].plot(t, res / 60 + j, color=C_V, lw=0.7)
        ax[0].plot(h["pressure_upstroke_s"], j, "|", color=C_NA, ms=8)
        ax[0].text(-0.21, j, r["record"], fontsize=6, va="center", ha="right")
    ax[0].axvline(0, color="#999", lw=0.5)
    ax[0].set_yticks([])
    ax[0].set_xlabel("s after the R peak")
    ax[0].set_title("recorded EEG locked to its own heartbeat, heart's field removed\n"
                    "(trace in units of its standard error / 60; red tick: arterial pressure upstroke)", fontsize=8)
    raw = [r["EEG"]["heart"]["in_phase_share_of_pair"] for r in R]
    res = [r["EEG_minus_field"]["heart"]["in_phase_share_of_pair"] for r in R]
    y = np.arange(len(R))
    ax[1].barh(y + 0.2, raw, 0.4, color="#e0a800", label="EEG as recorded")
    ax[1].barh(y - 0.2, res, 0.4, color=C_P, label="heart's field removed")
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([r["record"] for r in R], fontsize=6)
    ax[1].set_xlabel("share of the heart-locked pair read in phase, r0^2 / (r0^2 + r90^2)")
    ax[1].axvline(np.median(res), color=C_P, lw=0.8, ls="--")
    ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_recorded_eeg_heart.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    which = sys.argv[1:] or ["cell", "body", "profiles", "control", "recorded"]
    if "cell" in which:
        fig_cell()
    if "body" in which:
        fig_body()
    if "profiles" in which:
        fig_profiles()
    if "control" in which:
        fig_control()
    if "recorded" in which:
        fig_recorded()
