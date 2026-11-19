"""The NEURON cell (frozen ions) and NEST's hh_cond_exp_traub are one cell: same input, same membrane.

Author of the programme: Abhijit Singh, 19 November 2026.

Both receive the same excitatory and inhibitory spike times (NEST: spike_generator with delay 0.1 ms; NEURON: the
same times shifted by 0.1 ms). Compared: the voltage traces sample by sample, and the spike times.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell import NEST_UNIT, Cell, poisson_times  # noqa: E402


def run(t_stop=5000.0, rate_e=500.0, rate_i=400.0, w_e=6.0, w_i=20.0, seed=1):
    rng = np.random.default_rng(seed)
    te = poisson_times(rate_e, t_stop - 1, rng)
    ti = poisson_times(rate_i, t_stop - 1, rng)

    import nest
    nest.set_verbosity("M_ERROR")
    nest.ResetKernel()
    nest.resolution = 0.1
    n = nest.Create("hh_cond_exp_traub", params={"V_m": NEST_UNIT["E_L"]})
    ge = nest.Create("spike_generator", params={"spike_times": te})
    gi = nest.Create("spike_generator", params={"spike_times": ti})
    nest.Connect(ge, n, syn_spec={"weight": w_e, "delay": 0.1})
    nest.Connect(gi, n, syn_spec={"weight": -w_i, "delay": 0.1})
    mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 0.1})
    sr = nest.Create("spike_recorder")
    nest.Connect(mm, n)
    nest.Connect(n, sr)
    nest.Simulate(t_stop)
    ev = mm.get("events")
    t_nest, v_nest = ev["times"], ev["V_m"]
    sp_nest = sr.get("events")["times"]

    c = Cell("frozen")
    c.drive(te + 0.1, w_e, ti + 0.1, w_i)
    rec = c.record(0.1, currents=False)
    out = c.run(t_stop)
    t_nrn, v_nrn = out["t"], out["v"]
    # align on NEST's grid (NEST records at the end of each step)
    v_on = np.interp(t_nest, t_nrn, v_nrn)
    sub = v_nest < NEST_UNIT["V_T"] + 10          # compare away from spike peaks
    # NEST stamps a spike at the step where V has started to fall above V_T + 30; NEURON at the upward crossing of
    # V_T + 30. Compare peak times of V in both, found on the same grid.
    def peaks(t, v):
        i = np.nonzero((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]) & (v[1:-1] > NEST_UNIT["V_T"] + 30))[0] + 1
        return t[i]
    pk_nest = peaks(t_nest, v_nest)
    pk_nrn = peaks(t_nrn, v_nrn)
    m = min(len(pk_nest), len(pk_nrn))
    res = {
        "input": {"rate_e_Hz": rate_e, "rate_i_Hz": rate_i, "w_e_nS": w_e, "w_i_nS": w_i, "t_ms": t_stop},
        "spikes_nest": int(len(sp_nest)), "peaks_nest": int(len(pk_nest)), "peaks_neuron": int(len(pk_nrn)),
        "peak_time_diff_ms_max": float(np.max(np.abs(pk_nest[:m] - pk_nrn[:m]))) if m else None,
        "peak_time_diff_ms_median": float(np.median(np.abs(pk_nest[:m] - pk_nrn[:m]))) if m else None,
        "v_diff_subthreshold_mV_max": float(np.max(np.abs(v_on[sub] - v_nest[sub]))),
        "v_diff_subthreshold_mV_rms": float(np.sqrt(np.mean((v_on[sub] - v_nest[sub]) ** 2))),
        "rest_mV": float(c.rest["V0"]),
    }
    return res


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=1))
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "cell_check_nest.json"), "w") as f:
        json.dump(r, f, indent=1)
