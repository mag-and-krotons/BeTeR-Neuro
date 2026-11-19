"""The excitable cell as a pair exchange, in NEURON.

Author of the programme: Abhijit Singh, 19 November 2026.

One cell, one compartment, with the same membrane as NEST's hh_cond_exp_traub unit (area 20 000 um^2, C = 200 pF,
Traub-Miles m^3 h sodium and n^4 potassium, rate functions copied from NEST), so that the network unit of body/ is
this cell. Two ways to hold the cell:

* frozen: concentrations and reversal potentials fixed, one leak (E_L = -60 mV, g_L = 10 nS). This is NEST's unit
  exactly; check_against_nest() runs both on the same input.
* live: sodium and potassium tracked inside the cell; the leak split into its sodium and potassium members; the pump
  (3 Na out : 2 K in) closes the exchange. The split and the pump capacity are not chosen: they are solved from the
  closure of the resting state (close_rest), at NEST's resting potential and with NEST's total leak conductance.

Nothing in this file is fitted to a firing rate.
"""
from __future__ import annotations

import math
import os

import numpy as np
from neuron import h

HERE = os.path.dirname(os.path.abspath(__file__))
_lib = os.path.join(HERE, "x86_64", ".libs", "libnrnmech.so")
if not os.path.exists(_lib):
    _lib = os.path.join(HERE, "x86_64", "libnrnmech.so")
if not hasattr(h, "traub"):
    h.nrn_load_dll(_lib)
h.load_file("stdrun.hoc")

# ---------------------------------------------------------------- the NEST unit (hh_cond_exp_traub defaults)
NEST_UNIT = dict(C_m=200.0, g_L=10.0, E_L=-60.0, g_Na=20000.0, g_K=6000.0, E_Na=50.0, E_K=-90.0, V_T=-63.0,
                 E_ex=0.0, E_in=-80.0, tau_syn_ex=5.0, tau_syn_in=10.0, t_ref=2.0)
AREA_UM2 = 20000.0                          # C_m = 200 pF at 1 uF/cm2
AREA_CM2 = AREA_UM2 * 1e-8
DIAM = math.sqrt(AREA_UM2 / math.pi)        # cylinder with L = diam: area = pi d^2
CELSIUS = 36.0
NAO, KO = 145.0, 4.0                        # bath (mM)
FARADAY = 96485.33212
R_GAS = 8.314462618


def rt_over_f() -> float:
    return R_GAS * (CELSIUS + 273.15) / FARADAY * 1e3          # mV


def inside_for(e_rev: float, outside: float) -> float:
    """The inside concentration that gives reversal e_rev (mV) for a monovalent cation."""
    return outside / math.exp(e_rev / rt_over_f())


NAI0 = inside_for(NEST_UNIT["E_Na"], NAO)   # 22.2 mM at 36 C
KI0 = inside_for(NEST_UNIT["E_K"], KO)      # 117 mM


def ns_to_s_cm2(g_ns: float) -> float:
    return g_ns * 1e-9 / AREA_CM2


def pa_to_ma_cm2(i_pa: float) -> float:
    return i_pa * 1e-9 / AREA_CM2


def ma_cm2_to_pa(i: float) -> float:
    return i * AREA_CM2 * 1e9


# ---------------------------------------------------------------- Traub rates (same as the mod file / NEST)
def _xexp(x, y):
    x = np.asarray(x, float)
    small = np.abs(x / y) < 1e-6
    out = np.empty_like(x)
    out[small] = y * (1 - x[small] / y / 2)
    out[~small] = x[~small] / (np.exp(x[~small] / y) - 1)
    return out


def traub_rates(v, vt=NEST_UNIT["V_T"]):
    u = np.asarray(v, float) - vt
    an = 0.032 * _xexp(15 - u, 5)
    bn = 0.5 * np.exp((10 - u) / 40)
    am = 0.32 * _xexp(13 - u, 4)
    bm = 0.28 * _xexp(u - 40, 5)
    ah = 0.128 * np.exp((17 - u) / 18)
    bh = 4 / (1 + np.exp((40 - u) / 5))
    return am, bm, ah, bh, an, bn


def traub_steady(v):
    am, bm, ah, bh, an, bn = traub_rates(v)
    return am / (am + bm), ah / (ah + bh), an / (an + bn)


def hh_currents_pa(v, e_na=NEST_UNIT["E_Na"], e_k=NEST_UNIT["E_K"]):
    """Steady sodium and potassium currents of the Traub channels at v (pA, outward positive; nS x mV = pA)."""
    m, hh, n = traub_steady(v)
    return (NEST_UNIT["g_Na"] * m ** 3 * hh * (v - e_na), NEST_UNIT["g_K"] * n ** 4 * (v - e_k))


def frozen_steady_current(v):
    """Net steady outward current (pA) of the frozen unit at v. Where it is negative the membrane is pushed up."""
    ina, ik = hh_currents_pa(v)
    return NEST_UNIT["g_L"] * (v - NEST_UNIT["E_L"]) + ina + ik


def leak_members() -> tuple:
    """NEST's single leak (g_L, E_L) as its two members: g_NaL E_Na + g_KL E_K = g_L E_L, g_NaL + g_KL = g_L."""
    g, el, ena, ek = NEST_UNIT["g_L"], NEST_UNIT["E_L"], NEST_UNIT["E_Na"], NEST_UNIT["E_K"]
    g_nal = g * (el - ek) / (ena - ek)
    return g_nal, g - g_nal


def close_rest(kna: float = 10.0, kk: float = 1.5) -> dict:
    """The resting state as a closed exchange, nothing chosen. The leak members are NEST's leak split (leak_members);
    the concentrations are those that give NEST's E_Na and E_K. At rest sodium in and potassium out must be returned
    by the pump at 3:2:

        sodium:    g_NaL (V0 - E_Na) + I_Na,HH(V0) + 3 I_p = 0
        potassium: g_KL  (V0 - E_K)  + I_K,HH(V0)  - 2 I_p = 0

    Two equations in (V0, I_p). Eliminating I_p: 2 [sodium] + 3 [potassium] = 0, a root in V0 (bisection). Without the
    channels this is V0 = (2 g_NaL E_Na + 3 g_KL E_K) / (2 g_NaL + 3 g_KL), the 3:2 resting potential (the linear form of
    Mullins and Noda's 1963 equation). The pump capacity follows from I_p and the site occupancies."""
    g_nal, g_kl = leak_members()
    ena, ek = NEST_UNIT["E_Na"], NEST_UNIT["E_K"]

    def na_member(v):
        return g_nal * (v - ena) + hh_currents_pa(v)[0]

    def k_member(v):
        return g_kl * (v - ek) + hh_currents_pa(v)[1]

    def f(v):
        return 2 * na_member(v) + 3 * k_member(v)
    lo, hi = -90.0, -63.0              # the lower root; the upper one (near -57.5 mV) is the unstable threshold
    assert f(lo) * f(hi) < 0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    v0 = 0.5 * (lo + hi)
    i_p = k_member(v0) / 2.0
    occ = (NAI0 / (NAI0 + kna)) ** 3 * (KO / (KO + kk)) ** 2
    linear_v0 = (2 * g_nal * ena + 3 * g_kl * ek) / (2 * g_nal + 3 * g_kl)
    return {"V0": float(v0), "V0_without_channels": float(linear_v0), "g_NaL_nS": g_nal, "g_KL_nS": g_kl,
            "I_p_pA": float(i_p), "occupancy": occ, "I_max_pA": float(i_p / occ), "kna": kna, "kk": kk,
            "E_L_effective": float(NEST_UNIT["E_L"] - i_p / NEST_UNIT["g_L"]),
            "na_balance_pA": float(na_member(v0) + 3 * i_p), "k_balance_pA": float(k_member(v0) - 2 * i_p)}


# ---------------------------------------------------------------- the cell
class Cell:
    """One compartment. ions='frozen' is NEST's unit; ions='live' tracks Na and K with the pump closing them."""

    def __init__(self, ions: str = "frozen", rest: dict | None = None):
        self.ions = ions
        h.celsius = CELSIUS
        self.soma = s = h.Section(name=f"soma_{id(self)}")
        s.L = s.diam = DIAM
        s.cm = 1.0
        s.nseg = 1
        s.Ra = 100
        s.insert("traub")
        for seg in s:
            seg.traub.gnabar = ns_to_s_cm2(NEST_UNIT["g_Na"])
            seg.traub.gkbar = ns_to_s_cm2(NEST_UNIT["g_K"])
            seg.traub.vt = NEST_UNIT["V_T"]
        h.nao0_na_ion = NAO
        h.ko0_k_ion = KO
        h.nai0_na_ion = NAI0
        h.ki0_k_ion = KI0
        if ions == "frozen":
            s.insert("pas")
            for seg in s:
                seg.pas.g = ns_to_s_cm2(NEST_UNIT["g_L"])
                seg.pas.e = NEST_UNIT["E_L"]
            h.ion_style("na_ion", 1, 0, 0, 0, 0, sec=s)      # concentrations and reversals held
            h.ion_style("k_ion", 1, 0, 0, 0, 0, sec=s)
            s.ena = NEST_UNIT["E_Na"]
            s.ek = NEST_UNIT["E_K"]
            self.rest = {"V0": NEST_UNIT["E_L"]}      # NEST's initial V_m; this unit has no resting state
        else:
            self.rest = rest or close_rest()
            s.insert("nakleak")
            s.insert("nakpump")
            s.insert("ionacc")
            for seg in s:
                seg.nakleak.gnal = ns_to_s_cm2(self.rest["g_NaL_nS"])
                seg.nakleak.gkl = ns_to_s_cm2(self.rest["g_KL_nS"])
                seg.nakpump.imax = pa_to_ma_cm2(self.rest["I_max_pA"])
                seg.nakpump.kna = self.rest["kna"]
                seg.nakpump.kk = self.rest["kk"]
            h.ion_style("na_ion", 3, 2, 1, 1, 1, sec=s)      # nai a state, ena from Nernst every step
            h.ion_style("k_ion", 3, 2, 1, 1, 1, sec=s)
        self.v_init = self.rest["V0"]
        self.syn_e = h.ExpSyn(s(0.5))
        self.syn_e.tau, self.syn_e.e = NEST_UNIT["tau_syn_ex"], NEST_UNIT["E_ex"]
        self.syn_i = h.ExpSyn(s(0.5))
        self.syn_i.tau, self.syn_i.e = NEST_UNIT["tau_syn_in"], NEST_UNIT["E_in"]
        self._keep = []

    def drive(self, times_e, w_e_ns, times_i=None, w_i_ns=0.0):
        """Deliver presynaptic spike times (ms) to the excitatory and inhibitory synapses (weights in nS)."""
        for times, syn, w in ((times_e, self.syn_e, w_e_ns), (times_i, self.syn_i, w_i_ns)):
            if times is None or len(times) == 0:
                continue
            nc = h.NetCon(None, syn)
            nc.weight[0] = w * 1e-3                       # nS -> uS
            nc.delay = 0
            tt = np.asarray(times, float)

            def _queue(nc=nc, tt=tt):
                for t in tt:
                    nc.event(float(t))
            self._keep += [nc, h.FInitializeHandler(_queue), _queue]

    def record(self, dt_rec: float = 0.1, currents: bool = True) -> dict:
        s = self.soma(0.5)
        rec = {"t": h.Vector().record(h._ref_t, dt_rec), "v": h.Vector().record(s._ref_v, dt_rec)}
        if currents:
            rec["ina_hh"] = h.Vector().record(s._ref_inat_traub, dt_rec)
            rec["ik_hh"] = h.Vector().record(s._ref_ikt_traub, dt_rec)
            rec["ie"] = h.Vector().record(self.syn_e._ref_i, dt_rec)
            rec["ii"] = h.Vector().record(self.syn_i._ref_i, dt_rec)
            if self.ions == "live":
                rec["ina_l"] = h.Vector().record(s._ref_inal_nakleak, dt_rec)
                rec["ik_l"] = h.Vector().record(s._ref_ikl_nakleak, dt_rec)
                rec["ip"] = h.Vector().record(s._ref_ip_nakpump, dt_rec)
                rec["nai"] = h.Vector().record(s._ref_nai, dt_rec)
                rec["ki"] = h.Vector().record(s._ref_ki, dt_rec)
            else:
                rec["il"] = h.Vector().record(s._ref_i_pas, dt_rec)
        nc = h.NetCon(s._ref_v, None, sec=self.soma)
        nc.threshold = NEST_UNIT["V_T"] + 30.0
        self.spikes = h.Vector()
        nc.record(self.spikes)
        self._keep.append(nc)
        self._rec = rec
        return rec

    def run(self, tstop: float, atol: float = 1e-7, rtol: float = 1e-9):
        cv = h.CVode()
        cv.active(1)
        cv.atol(atol)
        cv.rtol(rtol)
        if self.ions == "live":
            cv.atolscale("nai", 1e-2)
            cv.atolscale("ki", 1e-1)
        h.finitialize(self.v_init)
        cv.re_init()
        h.continuerun(tstop)
        out = {k: np.array(v) for k, v in self._rec.items()}
        out["spikes"] = np.array(self.spikes)
        return out


def poisson_times(rate_hz, t_stop_ms, rng, t_start_ms=0.0, grid=0.1):
    """A homogeneous Poisson train on NEST's 0.1 ms grid (rate may be an array of instantaneous rates on grid)."""
    """Coincident events in one grid step are kept (repeated times), as NEST delivers them: each adds one weight."""
    if np.isscalar(rate_hz):
        n = rng.poisson(rate_hz * (t_stop_ms - t_start_ms) * 1e-3)
        t = np.sort(rng.uniform(t_start_ms, t_stop_ms, n))
        t = np.round(t / grid) * grid
    else:
        rate = np.asarray(rate_hz, float)
        counts = rng.poisson(rate * grid * 1e-3)
        t = np.repeat(t_start_ms + np.arange(rate.size) * grid, counts)
    return t[t > grid]
