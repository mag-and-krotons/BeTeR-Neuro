"""The cell's exchange, read whole: rest, one spike, the pump's return, and the cell under a carrier.

Author of the programme: Abhijit Singh, 19 November 2026.

Protocols (each writes its part of results/cell_exchange.json):

  rest      the live cell left alone for 120 s: does the closed rest stay closed (V, Na_i, K_i)?
  spike     one spike, charges of every member integrated: sodium in (channels, leak, pump), potassium out
            (channels, leak, pump), stimulus, C dV. The voltage is what the members leave uncancelled.
  train     20 spikes, then silence: how the pump member rises with the sodium the spikes brought in and how it
            returns. Its time constant is set against the one derived from the closure (tau_derived).
  carrier   the cell driven by input whose rate turns with a carrier at f (0.0125 to 2 Hz); every member read at f
            as a complex amplitude (in phase with the carrier, and in quadrature).

Units: pA, nS, mV, ms, pC, mM.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from neuron import h

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell import (AREA_CM2, DIAM, FARADAY, KI0, KO, NAI0, NEST_UNIT, Cell, close_rest,  # noqa: E402
                  ma_cm2_to_pa, poisson_times, rt_over_f)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "cell_exchange.json")
C_PF = NEST_UNIT["C_m"]
VOL_L = math.pi / 4 * DIAM ** 3 * 1e-15                 # um^3 -> L
PC_PER_MM = VOL_L * 1e-3 * FARADAY * 1e12                # charge (pC) that changes the inside by 1 mM


def tau_derived(rest: dict) -> dict:
    """The linear return of the inside sodium after a small load, from the closure alone.

    d Na_i/dt = -(I_NaL + I_Na,HH + 3 I_p) / (F vol). Near rest, the sodium members change with Na_i through the
    pump's three sodium sites (d(3 I_p)/d Na_i = 3 I_p * 3 kna / (Na_i (Na_i + kna))) and through E_Na in the sodium
    leak (d I_NaL/d Na_i = g_NaL RT/(F Na_i)). tau = F vol / (sum)."""
    ip, kna = rest["I_p_pA"], rest["kna"]
    d_pump = 3 * ip * 3 * kna / (NAI0 * (NAI0 + kna))                 # pA per mM
    d_leak = rest["g_NaL_nS"] * rt_over_f() / NAI0                    # nS * mV / mM = pA/mM
    tau_s = PC_PER_MM / (d_pump + d_leak)                              # pC / pA = s
    return {"pC_per_mM": PC_PER_MM, "dI_pump3_dNa_pA_per_mM": d_pump, "dI_NaL_dNa_pA_per_mM": d_leak,
            "tau_s": tau_s, "corner_Hz": 1 / (2 * math.pi * tau_s),
            "dI_pump_net_dNa_pA_per_mM": d_pump / 3}


def tau_full(rest: dict) -> dict:
    """The return of the inside ions with every member kept: Na_i and K_i together, the voltage at each instant at
    the value where the membrane's net current is zero (it settles in milliseconds, the ions in minutes).
    The Jacobian of (dNa_i/dt, dK_i/dt) at the closed rest, by central differences. Its eigenvalues give the
    return time and what is conserved."""
    from cell import KO as _KO, NAO, traub_steady
    g_nal, g_kl, imax = rest["g_NaL_nS"], rest["g_KL_nS"], rest["I_max_pA"]
    kna, kk = rest["kna"], rest["kk"]
    rtf = rt_over_f()

    def currents(v, nai, ki):
        ena, ek = rtf * math.log(NAO / nai), rtf * math.log(_KO / ki)
        m, hh, n = traub_steady(v)
        ip = imax * (nai / (nai + kna)) ** 3 * (_KO / (_KO + kk)) ** 2
        return (g_nal * (v - ena) + NEST_UNIT["g_Na"] * m ** 3 * hh * (v - ena) + 3 * ip,
                g_kl * (v - ek) + NEST_UNIT["g_K"] * n ** 4 * (v - ek) - 2 * ip)

    def v_q(nai, ki):
        lo, hi = -90.0, -63.0
        f = lambda v: sum(currents(v, nai, ki))           # noqa: E731
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if f(lo) * f(mid) <= 0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def rhs(x):
        ina, ik = currents(v_q(*x), *x)
        return np.array([-ina / PC_PER_MM, -ik / PC_PER_MM])
    x0 = np.array([NAI0, KI0])
    jac = np.zeros((2, 2))
    for j in range(2):
        d = np.zeros(2)
        d[j] = 1e-4
        jac[:, j] = (rhs(x0 + d) - rhs(x0 - d)) / 2e-4
    ev, evec = np.linalg.eig(jac)
    order = np.argsort(ev.real)
    ev, evec = ev[order].real, evec[:, order].real
    return {"jacobian_per_s": jac.tolist(), "eigenvalues_per_s": ev.tolist(),
            "return_tau_s": float(-1 / ev[0]), "return_mode_NaK": (evec[:, 0] / np.abs(evec[:, 0]).max()).tolist(),
            "conserved_mode_NaK": (evec[:, 1] / np.abs(evec[:, 1]).max()).tolist(),
            "second_eigenvalue_per_s": float(ev[1])}


def integrate(t_ms, y):
    return float(np.trapezoid(y, t_ms))


# ------------------------------------------------------------------------------------------------- rest
def proto_rest(rest):
    c = Cell("live", rest)
    c.record(1.0)
    out = c.run(120_000.0, atol=1e-8)
    return {"V_start": float(out["v"][0]), "V_end": float(out["v"][-1]),
            "V_max_drift_mV": float(np.max(np.abs(out["v"] - out["v"][0]))),
            "Na_i_drift_mM": float(np.max(np.abs(out["nai"] - NAI0))),
            "K_i_drift_mM": float(np.max(np.abs(out["ki"] - KI0))),
            "pump_pA": float(ma_cm2_to_pa(out["ip"][-1])), "spikes": int(len(out["spikes"]))}


# ------------------------------------------------------------------------------------------------- one spike
def proto_spike(rest, ions):
    """live: one spike evoked from the closed rest by 4 pC (4 nA for 1 ms), read until V returns to rest.
    frozen: NEST's unit has no rest; it fires on its own. One whole cycle of it is read, trough to trough."""
    c = Cell(ions, rest)
    amp = 4.0 if ions == "live" else 0.0
    stim = h.IClamp(c.soma(0.5))
    stim.delay, stim.dur, stim.amp = 50.0, 1.0, amp
    c._keep.append(stim)
    c.record(0.005)
    i_stim = h.Vector().record(stim._ref_i, 0.005)
    out = c.run(400.0 if ions == "live" else 1000.0, atol=1e-9, rtol=1e-10)
    t, v = out["t"], out["v"]
    istim = np.array(i_stim) * 1e3
    to_pa = ma_cm2_to_pa
    if ions == "live":
        i_on = np.searchsorted(t, 50.0)
        v0 = v[i_on - 1]
        i_pk = i_on + int(np.argmax(v[i_on:i_on + 4000]))
        after = np.nonzero(v[i_pk:] <= v0)[0]
        if len(after) == 0:
            return {"error": "no return to the start voltage", "V_peak": float(v[i_pk])}
        i_back = i_pk + int(after[0])
        i_start = i_on
    else:
        sp = out["spikes"]
        sp = sp[sp > 300]
        i1, i2 = np.searchsorted(t, sp[0]), np.searchsorted(t, sp[1])
        i3 = np.searchsorted(t, sp[2])
        i_start = i1 + int(np.argmin(v[i1:i2]))               # trough before the second spike
        i_back = i2 + int(np.argmin(v[i2:i3]))                # next trough
        i_pk = i_start + int(np.argmax(v[i_start:i_back]))
        v0 = v[i_start]
    sl = slice(i_start, i_back + 1)
    tt = t[sl]
    q = {"Na_channels_in": -integrate(tt, to_pa(out["ina_hh"][sl])),
         "K_channels_out": integrate(tt, to_pa(out["ik_hh"][sl])),
         "stimulus_in": integrate(tt, istim[sl])}
    if ions == "live":
        q["Na_leak_in"] = -integrate(tt, to_pa(out["ina_l"][sl]))
        q["K_leak_out"] = integrate(tt, to_pa(out["ik_l"][sl]))
        q["pump_net_out"] = integrate(tt, to_pa(out["ip"][sl]))
    else:
        q["leak_out"] = integrate(tt, to_pa(out["il"][sl]))
    q = {k: v_ * 1e-3 for k, v_ in q.items()}                 # pA*ms = fC -> pC
    dv = float(v[i_pk] - v0)
    inward = q["Na_channels_in"] + q["stimulus_in"] + q.get("Na_leak_in", 0.0)
    outward = q["K_channels_out"] + q.get("K_leak_out", 0.0) + q.get("pump_net_out", 0.0) + q.get("leak_out", 0.0)
    cdv = C_PF * float(v[i_back] - v0) * 1e-3
    res = {"read_over": "evoked spike, rest to rest" if ions == "live" else "one spontaneous cycle, trough to trough",
           "V_start": float(v0), "V_peak": float(v[i_pk]), "amplitude_mV": dv,
           "window_ms": [float(t[i_start]), float(t[i_back])], "charges_pC": q,
           "inward_minus_outward_pC": inward - outward, "C_dV_over_window_pC": cdv,
           "identity_residual_relative": abs((inward - outward) - cdv) / q["Na_channels_in"],
           "rho": q["Na_channels_in"] / (C_PF * dv * 1e-3),
           "voltage_reads_fraction_of_Na_charge": (C_PF * dv * 1e-3) / q["Na_channels_in"],
           "Na_load_mM": q["Na_channels_in"] / PC_PER_MM}
    if ions == "frozen":
        res["period_ms"] = float(sp[1] - sp[0])
    return res


# ------------------------------------------------------------------------------------------------- train
def proto_train(rest, n_spikes=20, rate_hz=20.0):
    c = Cell("live", rest)
    for k in range(n_spikes):
        st = h.IClamp(c.soma(0.5))
        st.delay, st.dur, st.amp = 1000.0 + k * 1000.0 / rate_hz, 1.0, 4.0
        c._keep.append(st)
    c.record(100.0)
    out = c.run(4_000_000.0, atol=1e-9)                      # 4000 s: the return is slow
    t = out["t"] * 1e-3
    ip = ma_cm2_to_pa(out["ip"])
    nai = out["nai"]
    t_end = 1.0 + n_spikes / rate_hz
    k0 = np.searchsorted(t, t_end + 0.2)
    # fit the return: log(nai - nai_inf) linear in t, with nai_inf the end value
    y = nai[k0:] - NAI0
    good = y > 1e-4 * y[0]
    slope = np.polyfit(t[k0:][good], np.log(y[good]), 1)[0]
    y2 = ip[k0:] - rest["I_p_pA"]
    good2 = y2 > 1e-3 * y2[0]
    slope2 = np.polyfit(t[k0:][good2], np.log(y2[good2]), 1)[0]
    return {"spikes_counted": int(len(out["spikes"])), "Na_i_peak_rise_mM": float(nai.max() - NAI0),
            "K_i_fall_mM": float(KI0 - out["ki"].min()), "pump_rise_pA": float(ip.max() - rest["I_p_pA"]),
            "V_min_after_mV": float(out["v"][k0:].min()), "V_rest_mV": float(rest["V0"]),
            "tau_Na_return_s": float(-1 / slope), "tau_pump_return_s": float(-1 / slope2),
            "pump_rise_per_spike_pA": float((ip.max() - rest["I_p_pA"]) / n_spikes)}


# ------------------------------------------------------------------------------------------------- carrier
def complex_at(t_s, y, f):
    """Complex amplitude of y at frequency f over an integer number of cycles: 2/T * integral y e^{-i 2 pi f t}.
    Real part: in phase with the carrier cos(2 pi f t); imaginary part with the opposite sign: quadrature."""
    w = 2 * np.pi * f
    return 2 * np.trapezoid(y * np.exp(-1j * w * t_s), t_s) / (t_s[-1] - t_s[0])


def proto_carrier(rest, freqs, r0=None, m=0.5, ri=None, w_e=6.0, w_i=20.0, seed=3, ions="live",
                  cycles=10, settle_s=20.0, min_s=60.0):
    """Excitatory input rate r0 (1 + m cos 2 pi f t), inhibitory rate ri; both Poisson on NEST's grid."""
    rows = []
    for f in freqs:
        rng = np.random.default_rng(seed)
        dur_s = max(cycles / f, min_s)
        dur_s = math.ceil(dur_s * f) / f                      # integer cycles
        t_stop_ms = (settle_s + dur_s) * 1e3
        grid = np.arange(0, t_stop_ms, 0.1)
        rate = r0 * (1 + m * np.cos(2 * np.pi * f * grid * 1e-3))
        te = poisson_times(rate, t_stop_ms, rng)
        ti = poisson_times(ri, t_stop_ms - 1, rng)
        c = Cell(ions, rest)
        c.drive(te, w_e, ti, w_i)
        c.record(1.0)
        out = c.run(t_stop_ms, atol=1e-7)
        t = out["t"] * 1e-3
        k = t >= settle_s
        ts = t[k] - settle_s
        sp = out["spikes"] * 1e-3
        sp = sp[sp >= settle_s] - settle_s
        n = len(sp)
        mean_rate = n / dur_s
        # the spikes as turns on the carrier: rate modulation as a complex amplitude
        z_sp = 2 * np.sum(np.exp(-1j * 2 * np.pi * f * sp)) / dur_s if n else 0j
        row = {"f_Hz": f, "duration_s": dur_s, "mean_rate_Hz": mean_rate, "spikes": n,
               "spike_rate_amp_Hz": abs(z_sp), "spike_phase_deg": float(np.degrees(np.angle(z_sp))),
               "spike_vector_strength": abs(z_sp) / (2 * mean_rate) if n else 0.0}
        v = np.minimum(out["v"][k], NEST_UNIT["V_T"] + 0)      # subthreshold voltage (spikes clipped)
        zv = complex_at(ts, v, f)
        row.update({"V_amp_mV": abs(zv), "V_phase_deg": float(np.degrees(np.angle(zv)))})
        if ions == "live":
            na_in = -ma_cm2_to_pa(out["ina_hh"][k]) - ma_cm2_to_pa(out["ina_l"][k])     # sodium entering (pA)
            ip = ma_cm2_to_pa(out["ip"][k])
            z_na = complex_at(ts, na_in, f)
            z_p = complex_at(ts, ip, f)
            z_nai = complex_at(ts, out["nai"][k], f)
            lag = float(np.degrees(np.angle(z_p / z_na)))
            row.update({"Na_in_amp_pA": abs(z_na), "Na_in_phase_deg": float(np.degrees(np.angle(z_na))),
                        "pump_amp_pA": abs(z_p), "pump_phase_deg": float(np.degrees(np.angle(z_p))),
                        "pump_minus_Na_in_deg": lag, "Na_i_amp_mM": abs(z_nai),
                        "mean_Na_i_mM": float(np.mean(out["nai"][k])), "mean_pump_pA": float(np.mean(ip))})
        rows.append(row)
        print(json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in row.items()}),
              flush=True)
    return rows


def save(part, value):
    try:
        with open(OUT) as f:
            allres = json.load(f)
    except (OSError, ValueError):
        allres = {}
    allres[part] = value
    with open(OUT, "w") as f:
        json.dump(allres, f, indent=1)


if __name__ == "__main__":
    which = sys.argv[1:] or ["closure", "spike", "train", "rest"]
    rest = close_rest()
    if "closure" in which:
        save("closure", {"rest": rest, "tau_derived": tau_derived(rest), "tau_full": tau_full(rest),
                         "concentrations_mM":
                         {"Na_i": NAI0, "Na_o": 145.0, "K_i": KI0, "K_o": KO}, "diam_um": DIAM,
                         "area_cm2": AREA_CM2})
        print(json.dumps(tau_derived(rest), indent=1))
    if "spike" in which:
        r = {"frozen": proto_spike(rest, "frozen"), "live": proto_spike(rest, "live")}
        save("spike", r)
        print(json.dumps(r, indent=1))
    if "train" in which:
        r = proto_train(rest)
        save("train", r)
        print(json.dumps(r, indent=1))
    if "rest" in which:
        r = proto_rest(rest)
        save("rest", r)
        print(json.dumps(r, indent=1))
