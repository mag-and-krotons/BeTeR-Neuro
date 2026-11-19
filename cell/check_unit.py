"""The live NEURON cell and the NEST network unit (unit.py) on the same input, read as values and as phase.

Author of the programme: Abhijit Singh, 19 November 2026.

Input: excitatory Poisson rate r0 (1 + m cos 2 pi f t) and constant inhibitory rate, identical spike times to both
(NEST through spike_generators with delay 0.1 ms). Read: spike counts, spike-by-spike coincidence (within 1 ms),
and the spikes as a complex amplitude at the carrier f (in phase and quadrature) in both.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell import Cell, close_rest, poisson_times  # noqa: E402
from unit import unit_params  # noqa: E402


def run(f=0.25, r0=600.0, m=0.5, ri=500.0, w_e=6.0, w_i=20.0, t_s=120.0, seed=11):
    rng = np.random.default_rng(seed)
    t_ms = t_s * 1e3
    grid = np.arange(0, t_ms, 0.1)
    te = poisson_times(r0 * (1 + m * np.cos(2 * np.pi * f * grid * 1e-3)), t_ms - 1, rng)
    ti = poisson_times(ri, t_ms - 1, rng)
    import nest
    nest.set_verbosity("M_ERROR")
    nest.ResetKernel()
    nest.resolution = 0.1
    n = nest.Create("hh_cond_exp_traub", params=unit_params())
    ge = nest.Create("spike_generator", params={"spike_times": te})
    gi = nest.Create("spike_generator", params={"spike_times": ti})
    nest.Connect(ge, n, syn_spec={"weight": w_e, "delay": 0.1})
    nest.Connect(gi, n, syn_spec={"weight": -w_i, "delay": 0.1})
    sr = nest.Create("spike_recorder")
    nest.Connect(n, sr)
    nest.Simulate(t_ms)
    sp_nest = np.sort(sr.get("events")["times"])
    c = Cell("live", close_rest())
    c.drive(te + 0.1, w_e, ti + 0.1, w_i)
    c.record(1.0, currents=False)
    out = c.run(t_ms, atol=1e-7)
    sp_nrn = out["spikes"]
    # coincidences within 1 ms
    j = np.searchsorted(sp_nest, sp_nrn)
    d = np.minimum(np.abs(sp_nrn - sp_nest[np.clip(j - 1, 0, len(sp_nest) - 1)]),
                   np.abs(sp_nrn - sp_nest[np.clip(j, 0, len(sp_nest) - 1)]))

    def z(sp):
        return 2 * np.sum(np.exp(-1j * 2 * np.pi * f * sp * 1e-3)) / t_s
    zn, zr = z(sp_nest), z(sp_nrn)
    return {"f_Hz": f, "spikes_nest": int(len(sp_nest)), "spikes_neuron": int(len(sp_nrn)),
            "coincident_within_1ms": int(np.sum(d <= 1.0)), "median_offset_ms": float(np.median(d)),
            "nest_rate_mod_Hz_IQ": [float(zn.real), float(-zn.imag)],
            "neuron_rate_mod_Hz_IQ": [float(zr.real), float(-zr.imag)],
            "final_Na_i_rise_mM": None}


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=1))
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "cell_check_unit.json"), "w") as fh:
        json.dump(r, fh, indent=1)
