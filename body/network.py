"""The body read by its nervous system: organs (carriers) -> vagal afferents (turns) -> brainstem -> relay ->
cortex, with the front (the senses facing outward) entering the cortex beside it, and a down-going efferent output.

Author of the programme: Abhijit Singh, 19 November 2026.

Every neuron is the network unit derived from the live cell (cell/unit.py): NEST's hh_cond_exp_traub with the
leak's reversal moved by the pump's resting current (E_L' = -68.47 mV), conductance synapses (5 ms excitatory to
0 mV, 10 ms inhibitory to -80 mV). One transduction gain for every organ: the organs differ only in their carriers.

Populations (sizes are the model's construction, stated, not read):
  aff_baroA 20, aff_baroC 40   arterial stretch endings, myelinated and unmyelinated fibres
  aff_stretch 40               lung stretch endings (myelinated)
  aff_gastric 40               stomach wall tension endings (unmyelinated)
  nts_{cardio,resp,gastric}    brainstem nucleus of the solitary tract, three zones, 120 E + 30 I each
  relay 150 E                  parabrachial / visceral thalamus
  ctx 1600 E + 400 I           cortex, balanced E/I; E 0-399 insula (body), 400-799 sensory (front), rest association
  eff 40 E                     dorsal motor nucleus / nucleus ambiguus: the down-going member, read, not fed back

Delays are readings of time of flight (conduction velocity x path length):
  baro A 20 m/s x 0.25 m = 12.5 ms; baro C 1 m/s x 0.25 m = 250 ms; lung stretch 30 m/s x 0.3 m = 10 ms;
  gastric C 1 m/s x 0.5 m = 500 ms; NTS -> relay 10 ms; relay -> insula 5 ms; cortical 1.5 ms; NTS -> eff 2 ms;
  insula -> eff 15 ms.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "cell"))
sys.path.insert(0, HERE)
from unit import unit_params  # noqa: E402

import carriers as carriers_mod  # noqa: E402

CFG = {
    "dt": 0.1,
    "threads": 2,
    "w_e": 6.0, "w_i": 67.0,                     # nS (Brette et al. 2007 COBAHH weights for this unit)
    "aff": {"baroA": (20, "heart", 12.5), "baroC": (40, "heart", 250.0), "stretch": (40, "breath", 10.0),
            "gastric": (40, "stomach", 500.0)},
    "aff_gain_hz": 1200.0,                        # input rate to an ending at full drive (same for every organ)
    "aff_bg": (300.0, 500.0),                     # background E, I rates to afferents (Hz)
    "nts": {"cardio": ("baroA", "baroC"), "resp": ("stretch",), "gastric": ("gastric",)},
    "nts_size": (120, 30), "nts_in_from_aff": 10, "nts_rec": (20, 10), "nts_cross": 4,
    "relay_size": 150, "relay_in": 30, "ctx_size": (1600, 400), "ctx_K": (80, 20), "insula_in": 20,
    "eff_size": 40, "eff_in_nts": 10, "eff_in_insula": 10,
    "front_gain_hz": 800.0, "front_hold_s": 0.5,
    # background (the rest of the nervous system): set once so that every population fires on fluctuations at a
    # few Hz with its own inputs present (5 s runs; operating point of the construction, not a fit to any record)
    "bg": {"nts": (300.0, 500.0), "relay": (300.0, 500.0), "ctx": (150.0, 600.0), "eff": (250.0, 500.0)},
    "delays": {"nts_relay": 10.0, "relay_insula": 5.0, "ctx": 1.5, "nts_eff": 2.0, "insula_eff": 15.0,
               "local": 1.0},
    "multimeter": {"interval": 5.0, "n_per_group": 30},
}


def build(car, cfg, duration_s, seed=1, flat=False):
    """flat=True: every organ's drive replaced by its own mean over the run (same mean input to every ending, no
    carrier). The control that keeps every value of the input and removes only its phase."""
    import nest
    nest.set_verbosity("M_ERROR")
    nest.ResetKernel()
    nest.resolution = cfg["dt"]
    nest.local_num_threads = cfg["threads"]
    nest.rng_seed = seed
    up = unit_params()
    we, wi = cfg["w_e"], cfg["w_i"]
    rng = np.random.default_rng(seed)
    P = {}

    def pop(name, n):
        p = nest.Create("hh_cond_exp_traub", n, params=up)
        # start each neuron somewhere near rest, not all in step
        p.V_m = up["V_m"] + rng.uniform(-2, 2, n)
        P[name] = p
        return p

    def background(p, re, ri):
        ge = nest.Create("poisson_generator", params={"rate": re})
        gi = nest.Create("poisson_generator", params={"rate": ri})
        nest.Connect(ge, p, syn_spec={"weight": we, "delay": cfg["delays"]["local"]})
        nest.Connect(gi, p, syn_spec={"weight": -20.0, "delay": cfg["delays"]["local"]})

    # ---------------------------------------------------------------- the organs' drives into the endings
    t_ms = car["t"] * 1e3
    keep = t_ms < duration_s * 1e3
    step = 1                                                       # drives on a 1 ms grid
    for name, (n, organ, delay) in cfg["aff"].items():
        p = pop("aff_" + name, n)
        drive = car[organ]["drive"][keep][::step]
        if flat:
            drive = np.full_like(drive, drive.mean())
        rate = cfg["aff_gain_hz"] * drive
        times = t_ms[keep][::step] + cfg["dt"]
        gen = nest.Create("inhomogeneous_poisson_generator", n, params={"rate_times": np.round(times, 1),
                                                                      "rate_values": rate})
        nest.Connect(gen, p, "one_to_one", syn_spec={"weight": we, "delay": cfg["delays"]["local"]})
        background(p, *cfg["aff_bg"])

    # ---------------------------------------------------------------- brainstem
    ne, ni = cfg["nts_size"]
    for zone, sources in cfg["nts"].items():
        e = pop(f"nts_{zone}_E", ne)
        i = pop(f"nts_{zone}_I", ni)
        for src in sources:
            n_src, organ, delay = cfg["aff"][src]
            k = min(cfg["nts_in_from_aff"], n_src)
            for tgt in (e, i):
                nest.Connect(P["aff_" + src], tgt, {"rule": "fixed_indegree", "indegree": k},
                             syn_spec={"weight": we, "delay": delay})
        ke, ki = cfg["nts_rec"]
        for tgt in (e, i):
            nest.Connect(e, tgt, {"rule": "fixed_indegree", "indegree": ke},
                         syn_spec={"weight": we, "delay": cfg["delays"]["local"]})
            nest.Connect(i, tgt, {"rule": "fixed_indegree", "indegree": ki},
                         syn_spec={"weight": -wi, "delay": cfg["delays"]["local"]})
        background(e, *cfg["bg"]["nts"])
        background(i, *cfg["bg"]["nts"])
    zones = list(cfg["nts"])
    for a in zones:
        for b in zones:
            if a != b:
                nest.Connect(P[f"nts_{a}_E"], P[f"nts_{b}_E"], {"rule": "fixed_indegree",
                                                                "indegree": cfg["nts_cross"]},
                             syn_spec={"weight": we, "delay": cfg["delays"]["local"]})
    nts_e = sum((P[f"nts_{z}_E"] for z in zones[1:]), P[f"nts_{zones[0]}_E"])

    # ---------------------------------------------------------------- relay and cortex
    relay = pop("relay", cfg["relay_size"])
    nest.Connect(nts_e, relay, {"rule": "fixed_indegree", "indegree": cfg["relay_in"]},
                 syn_spec={"weight": we, "delay": cfg["delays"]["nts_relay"]})
    background(relay, *cfg["bg"]["relay"])
    ce, ci = cfg["ctx_size"]
    ctx_e = pop("ctx_E", ce)
    ctx_i = pop("ctx_I", ci)
    ke, ki = cfg["ctx_K"]
    for tgt in (ctx_e, ctx_i):
        nest.Connect(ctx_e, tgt, {"rule": "fixed_indegree", "indegree": ke},
                     syn_spec={"weight": we, "delay": cfg["delays"]["ctx"]})
        nest.Connect(ctx_i, tgt, {"rule": "fixed_indegree", "indegree": ki},
                     syn_spec={"weight": -wi, "delay": cfg["delays"]["ctx"]})
        background(tgt, *cfg["bg"]["ctx"])
    insula = ctx_e[0:400]
    sensory = ctx_e[400:800]
    nest.Connect(relay, insula, {"rule": "fixed_indegree", "indegree": cfg["insula_in"]},
                 syn_spec={"weight": we, "delay": cfg["delays"]["relay_insula"]})

    # ---------------------------------------------------------------- the front: the senses facing outward
    hold = cfg["front_hold_s"]
    n_hold = int(np.ceil(duration_s / hold))
    levels = rng.uniform(0, 1, n_hold)
    front_times = np.round(np.arange(n_hold) * hold * 1e3 + cfg["dt"], 1)
    front = nest.Create("inhomogeneous_poisson_generator", 400,
                        params={"rate_times": front_times, "rate_values": cfg["front_gain_hz"] * levels})
    nest.Connect(front, sensory, "one_to_one", syn_spec={"weight": we, "delay": cfg["delays"]["local"]})

    # ---------------------------------------------------------------- the down-going member
    eff = pop("eff", cfg["eff_size"])
    nest.Connect(nts_e, eff, {"rule": "fixed_indegree", "indegree": cfg["eff_in_nts"]},
                 syn_spec={"weight": we, "delay": cfg["delays"]["nts_eff"]})
    nest.Connect(insula, eff, {"rule": "fixed_indegree", "indegree": cfg["eff_in_insula"]},
                 syn_spec={"weight": we, "delay": cfg["delays"]["insula_eff"]})
    background(eff, *cfg["bg"]["eff"])

    # ---------------------------------------------------------------- readings
    rec = {}
    for name, p in P.items():
        sr = nest.Create("spike_recorder")
        nest.Connect(p, sr)
        rec[name] = sr
    k = cfg["multimeter"]["n_per_group"]
    mm_groups = {"insula": insula[:k], "sensory": sensory[:k], "assoc": ctx_e[800:800 + k],
                 "ctx_I": ctx_i[:k], "nts_cardio_E": P["nts_cardio_E"][:k]}
    mms = {}
    for g, nodes in mm_groups.items():
        mm = nest.Create("multimeter", params={"record_from": ["V_m", "g_ex", "g_in"],
                                               "interval": cfg["multimeter"]["interval"]})
        nest.Connect(mm, nodes)
        mms[g] = mm
    gids = {name: [int(p[0].global_id), int(p[-1].global_id)] for name, p in P.items()}
    return {"pops": P, "spikes": rec, "mm": mms, "front_levels": levels, "front_times_ms": front_times,
            "gids": gids}


def run(duration_s=300.0, out_name="body_run", seed=1, cfg=None, flat=False):
    import nest
    cfg = cfg or CFG
    car = carriers_mod.load(duration_s=duration_s + 1.0)
    t0 = time.time()
    net = build(car, cfg, duration_s, seed=seed, flat=flat)
    t_build = time.time() - t0
    nest.Simulate(duration_s * 1e3)
    t_sim = time.time() - t0 - t_build
    out = {}
    for name, sr in net["spikes"].items():
        ev = sr.get("events")
        out[f"spk_{name}_t"] = ev["times"].astype(np.float32)
        out[f"spk_{name}_id"] = ev["senders"].astype(np.int32)
    for g, mm in net["mm"].items():
        ev = mm.get("events")
        order = np.lexsort((ev["times"], ev["senders"]))
        for key in ("times", "senders", "V_m", "g_ex", "g_in"):
            out[f"mm_{g}_{key}"] = ev[key][order].astype(np.float32 if key != "senders" else np.int32)
    out["front_levels"] = net["front_levels"]
    out["front_times_ms"] = net["front_times_ms"]
    for k in ("heart", "breath", "stomach"):
        out[f"car_{k}_phase"] = car[k]["phase"][::10].astype(np.float64)       # 100 Hz
        out[f"car_{k}_drive"] = car[k]["drive"][::10].astype(np.float32)
    out["car_t"] = car["t"][::10]
    out["heart_events_s"] = car["heart"]["events_s"]
    path = os.path.join(ROOT, "results", out_name + ".npz")
    np.savez_compressed(path, **out)
    meta = {"duration_s": duration_s, "seed": seed, "flat_control": flat, "build_s": t_build, "sim_s": t_sim, "cfg": cfg,
            "gids": net["gids"], "carrier_sources": {k: car[k]["source"] for k in ("heart", "breath", "stomach")},
            "carrier_frequencies_Hz": carriers_mod.frequencies(car),
            "rates_Hz": {name: float(len(out[f"spk_{name}_t"]) / (len(net["pops"][name]) * duration_s))
                         for name in net["pops"]}}
    with open(os.path.join(ROOT, "results", out_name + ".json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return meta


if __name__ == "__main__":
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    name = sys.argv[2] if len(sys.argv) > 2 else "body_run"
    m = run(dur, name, flat=(len(sys.argv) > 3 and sys.argv[3] == "flat"))
    print(json.dumps({k: m[k] for k in ("sim_s", "rates_Hz", "carrier_frequencies_Hz")}, indent=1))
