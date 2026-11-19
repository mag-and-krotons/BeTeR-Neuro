# beter-neuro: the body read at the horizon (NEURON + NEST)

Programme author: **Abhijit Singh**, 19 November 2026.

This is the neural model described in the Three-π record ("Intelligence, and a true neural model"). Each excitable
cell is an exchange between two members, sodium coming in and potassium going out, closed by the 3:2 pump. Each
organ is a carrier (heartbeat, breath, the stomach's slow wave), and the afferent spikes are turns riding on it.
Organs reach the brain by phase, with delay, mostly upward. The mind's reading is taken across all of them, with both
members kept (in phase and in quadrature), never by instantaneous correlation or firing rate alone.

It also takes the next step named in the Outlook of paper 05 (*The Balanced Pair in Excitable Tissue*): a spiking
network with conductance-based synapses, with its excitatory and inhibitory members recorded.

## What it establishes

1. **A cell rests only as a closed exchange.** NEST's standard unit has one leak and no pump, and it has no
   resting state: it fires on its own. Split the leak into its sodium and potassium members, close them with the
   3:2 pump, and the rest appears at −68.45 mV with nothing chosen. A spike's voltage reads 30% of the sodium that
   crossed; the rest is the exchange. The ions return as one Na↔K mode, and their sum is conserved.
2. **Where an organ is read is set by frequency × time of flight.** Heart (0.96 Hz) through C fibres: predicted
   114°, read in the brainstem at 108° and in the cortex at 116°. Breath and stomach arrive in phase. The fastest
   carrier lands in quadrature, and every slow fibre still arrives in phase on a slow carrier.
3. **The rate says the body is being read; the phase says which organ, where in its cycle, and by which fibres.**
   The same input held at its mean wipes out every locking.
4. **The same holds in people.** In all 18 recordings (84 h), the EEG is locked to its own heartbeat. The zero-delay
   reading is taken by the heart's own electric field. With that field removed, the brain's heart-locked signal is
   still there in all 18, and a median 69% of it sits in quadrature, where an instantaneous reading cannot see it.
   It arrives as more than one wave per beat, 0.09–0.34 s and later after the R peak. The model gave two arrivals
   per beat with 81% in quadrature. The data were not used to build the model.

Every result below is marked **checked** (computed here, reproducible from the script named), **source** (a fact
from outside) or **reading** (an interpretation, not a computation). Nothing is fitted to a recorded firing rate.

```
   front (senses, outward) ───────────────────────────────┐
                                                          ▼
 heart ─ R+PEP ─▶ baro endings ─A 12.5 ms / C 250 ms─▶ NTS cardio ─┐        ┌─▶ cortex (balanced E/I, 1600+400)
 breath ────────▶ lung endings ─A 10 ms─────────────▶ NTS resp ────┼─▶ relay ─┤     insula | sensory | assoc
 stomach ───────▶ wall endings ─C 500 ms────────────▶ NTS gastric ─┘ 10 ms   └─▶ 5 ms
                                                            └──────▶ efferent (down-going, read) ◀── insula 15 ms
 every neuron: the cell below (hh_cond_exp_traub, E_L' from the pump closure)
```

## 1. The cell (NEURON): `cell/`

| result | value | mark |
|---|---|---|
| The NEURON cell (frozen ions) and NEST `hh_cond_exp_traub` on the same input, 5 s | 42/42 spikes at the same peak times; V differs by at most 0.031 mV (rms 0.0006) | checked, `check_nest.py` |
| NEST's default unit has **no resting state**: its steady current is inward at every V below −33.4 mV | it fires alone at 13.9 Hz (period 72.2 ms) | checked, `cell.frozen_steady_current`, `exchange.py spike` |
| NEST's single leak (10 nS, −60 mV) as its two members | g_NaL 2.14 nS, g_KL 7.86 nS (from E_L, E_Na, E_K; nothing chosen) | checked, `cell.leak_members` |
| Rest closed by the 3:2 pump (Na balance and K balance) | V0 = −68.445 mV (3:2 linear form: −68.462), pump 84.7 pA, I_max 488.6 pA | checked, `cell.close_rest` |
| The closed rest left alone for 120 s | V drifts 1.6e-9 mV; Na_i, K_i 2e-11 mM | checked, `exchange.py rest` |
| One spike from the closed rest | Na in 77.65 pC, K out 80.88 pC; charge identity holds to 1.2e-4 (sampling) | checked, `exchange.py spike` |
| What the voltage reads | C·ΔV is **30%** of the sodium charge (ρ = 3.32; frozen cycle ρ = 3.15) | checked |
| Return of the inside ions after 20 spikes | τ = 3106 s (simulated) against 3114 s (full linearisation of Na and K together) and 2908 s (Na side only) | checked, `exchange.py train`, `tau_full` |
| How the ions return | one mode, Na and K in equal and opposite steps (the exchange); the second mode is conserved (eigenvalue 2e-14 /s) | checked |
| The network unit: `hh_cond_exp_traub` with E_L' = E_L − I_p/g_L = −68.468 mV | the live cell and this unit, 120 s, about 10 Hz: 1161 of 1172 spikes within 1 ms; carrier reading (I, Q) 12.91, 1.02 Hz against 12.70, 1.04 Hz | checked, `unit.py`, `check_unit.py` |

The pump's slow member: in this 80 µm well-mixed cell, one spike raises Na_i by 0.002 mM and the pump by 0.007 pA,
returning in about 52 minutes. Over the carriers' seconds it is the constant that E_L' carries; over minutes of firing
(2.3 mM in 120 s at 10 Hz) it accounts for the 11 spikes in which the live cell and the unit differ.

## 2. The body read by the nervous system (NEST): `body/`

The heart and breath carriers are **recorded from one person on one clock**: ECG and respiration belt from
NeuroKit2's `bio_resting_8min_100hz` (heart 0.960 Hz, breath 0.134 Hz) (**source**). The stomach carrier (0.050 Hz) is
a slow wave of 3 cycles per minute. Delays are conduction velocity × path length (**source**,
readings of time of flight). Every organ uses the same transduction gain, so the organs differ only in their
carriers. The background was set once so that each population fires on fluctuations at a few Hz; that is the
operating point of the construction, not a fit. 300 s, about 3000 neurons, 2 threads, 32 min.

**Each population read against each carrier** (`results/body_run_reading.json`). |Z| is how strongly the spikes ride
the carrier, and the phase is where in the cycle they ride (0 = R peak, peak of the breath wave, slow-wave peak).
r0 is the in-phase reading (zero-lag correlation of the rate with the carrier) and r90 is its quadrature partner.
Every |Z| shown is beyond 200 time-shifted controls (p < 0.005) except the off-carrier entries of the afferent rows.

| population | rate (Hz) | heart: \|Z\| at phase | r0 / r90 | breath: \|Z\| at phase | r0 / r90 | stomach: \|Z\| at phase | r0 / r90 |
|---|---|---|---|---|---|---|---|
| aff_baroA | 12.0 | 0.928 at 61° | +0.260 / +0.486 | 0.077 at 40° | +0.025 / -0.004 | 0.016 at 99° | +0.001 / +0.004 |
| aff_stretch | 47.5 | 0.028 at 57° | +0.023 / +0.037 | 0.606 at 2° | +0.946 / -0.060 | 0.030 at 165° | -0.042 / -0.001 |
| aff_gastric | 41.9 | 0.003 at -22° | +0.004 / -0.003 | 0.036 at 85° | -0.016 / -0.029 | 0.672 at 1° | +0.949 / +0.006 |
| nts_cardio_E | 11.1 | 0.468 at 108° | -0.133 / +0.391 | 0.228 at 9° | +0.180 / -0.017 | 0.249 at 10° | +0.219 / +0.032 |
| nts_resp_E | 14.7 | 0.122 at 106° | -0.043 / +0.140 | 0.490 at 4° | +0.556 / -0.032 | 0.179 at 17° | +0.210 / +0.054 |
| nts_gastric_E | 15.3 | 0.130 at 115° | -0.061 / +0.126 | 0.172 at 13° | +0.161 / -0.020 | 0.582 at 10° | +0.626 / +0.105 |
| relay | 25.9 | 0.310 at 116° | -0.111 / +0.217 | 0.376 at 5° | +0.275 / -0.020 | 0.479 at 12° | +0.371 / +0.071 |
| ctx_insula | 17.8 | 0.281 at 116° | -0.067 / +0.130 | 0.348 at 5° | +0.168 / -0.013 | 0.427 at 13° | +0.218 / +0.046 |
| ctx_sensory | 13.6 | 0.111 at 116° | -0.032 / +0.062 | 0.171 at 14° | +0.093 / -0.009 | 0.148 at 19° | +0.091 / +0.024 |
| ctx_assoc | 7.7 | 0.222 at 116° | -0.051 / +0.097 | 0.288 at 7° | +0.130 / -0.011 | 0.324 at 14° | +0.157 / +0.036 |
| eff | 18.8 | 0.330 at 122° | -0.101 / +0.158 | 0.396 at 5° | +0.210 / -0.013 | 0.512 at 12° | +0.286 / +0.057 |

(baro C fires exactly as baro A: both endings feel the same arterial stretch, and they differ only in the delay to
the brainstem.)

### What the readings show

- **checked:** the heart reaches the cortex at **116°** of its cycle. The in-phase reading of the cortex gives r0 =
  −0.03 to −0.07: small, and opposite in sign to the coupling. It holds cos²116° = 19% of a locking whose full size
  is 0.11–0.28. Breath (5–14°) and stomach (10–19°) arrive almost in phase, so an in-phase reading gets nearly all
  of them.
- **checked:** these phases are what time of flight times frequency gives: heart via C fibres 114°, via A fibres
  32°; breath 0.5°; stomach via C fibres 9° (`flight_phase_deg`). The slowest fibres (stomach, 500 ms) arrive in
  phase because their carrier is slow. The fastest carrier (heart) lands in quadrature.
- **checked:** the single heart phase is itself an average. Over the cycle, NTS, relay, insula and efferent fire
  **twice per beat**: 0.16–0.19 s after the R peak (A fibres) and 0.39–0.42 s (C fibres). See `heart_profile` and
  fig. 3. One number (116°) reports neither arrival.
- **checked:** in every recorded group the excitatory and inhibitory members cancel 93.8–98.4%. The residual drives
  the firing (`members`), which is paper 05's balance in a conductance-based HH network.
- **checked:** the front (the sensory stimulus) is read by the sensory cortex at r = 0.27 with a 10 ms lag, and by the
  insula at 0.035. The body is read by the insula at |Z| 0.28–0.43. Each region reads mainly one side; the joint
  reading holds both.

### Control: each organ's drive held at its own mean (`body_flat`, same 300 s)

- **checked:** with the same mean input but no carrier, every locking falls to the level of the controls (all
  p > 0.05). The rates fall too: baro endings 12.0 → 2.5 Hz, insula 17.8 → 3.6 Hz, cortex E 11.7 → 4.6 Hz.
- **reading:** the threshold reads the carrier's peaks, so a drive with the same mean value gives a different mean
  rate. The rate registers that the body's carriers are present; which organ, where in its cycle and by which fibres
  are in the phase (fig. 4).

## 3. The same reading on a recorded body: `reading/recorded.py`

Data: MIT-BIH Polysomnographic Database (PhysioNet slpdb): **all 18 records**, 84 h in total. Each record has ECG,
invasive arterial pressure, one EEG lead and respiration from one body on one clock, at 250 Hz. Every .dat file
matches PhysioNet's published SHA-256. The heart's events are the database's own normal-beat annotations; our
R-peak detector agrees within 20 ms on 15 records, and misses on three with inverted ECG (slp04, slp48, slp03), which
is why the annotations are used. The EEG reading is exactly the model's: the first harmonic against the heart's phase
(in phase and quadrature), with 200 time-shifted controls. Stretches where the EEG saturates are excluded (98% of
samples kept in slp01a).

| result | value | mark |
|---|---|---|
| EEG locked to its own heartbeat | all 18 records, p < 0.005 against shifted controls | checked |
| The heart's own electric field in the EEG | reaches the scalp at zero delay, with the ECG's shape. Read in phase, the raw EEG gives a median in-phase share of 0.55 (0.01–1.00) | checked |
| With that field taken out (EEG minus its zero-delay ECG share, fitted over the whole record) | still locked in all 18 (p < 0.005); the median share read in phase falls to **0.31**. Most of what the brain carries of the heart is in quadrature | checked |
| When it arrives (heart-locked average with the field removed, extrema > 4 SE between 0.05 and 0.8 s) | largest features at 0.09–0.34 s after R; most records show two or three (for example slp01b 0.14 / 0.30 / 0.50 s, slp48 0.18 / 0.34 s, slp04 0.24 / 0.68 s) | checked, `heart_locked` |
| The arterial pressure upstroke (radial line) | 0.18–0.24 s after R | checked |
| EEG locked to breath | 17 of 18 records at p < 0.05, but small (\|Z\| 0.001–0.12) | checked |

- **reading:** the recorded EEG gives the same answer as the model. Its in-phase reading is dominated by the heart's
  own field, which carries no delay. What arrives later sits mostly in quadrature (model: cos²116° = 0.19; recorded:
  median 0.31), and it comes as more than one arrival after each beat. The numbers are set side by side here, not
  scored against each other: the model's delays are fibre readings, and an EEG lead adds its own filtering.
Run: `python3 reading/recorded.py read` (data in `data/slpdb`). Figure: `results/figures/fig5_recorded_eeg_heart.png`.

## Reproduce

```bash
cd cell && nrnivmodl mod && python3 check_nest.py && python3 exchange.py closure spike train rest && python3 check_unit.py
cd .. && python3 body/network.py 300 body_run && python3 body/network.py 300 body_flat flat
python3 reading/reading.py body_run && python3 reading/reading.py body_flat && python3 figures.py
```
Requires NEURON ≥ 9, NEST 3.10 (pip wheels used here), numpy, scipy, pandas, matplotlib; wfdb for `recorded.py`.
The raw spike files (`results/body_run.npz`, `body_flat.npz`, 150 MB each) are regenerated by the commands above.

## Files

`cell/mod/` traub.mod (= NEST's Traub rates), nakleak.mod (the two leak members), nakpump.mod (3:2 pump),
ionacc.mod (Na, K inside) · `cell/cell.py` the cell and the closure · `cell/exchange.py` rest, spike, train, carrier
protocols · `cell/unit.py` the derived NEST unit · `body/carriers.py` organ carriers · `body/network.py` the body–brain
network · `reading/reading.py` the complete reading · `reading/recorded.py` the recorded-body reading ·
`figures.py` · `results/` JSON and figures (`recorded_slpdb_reading.json` holds every record's readings and
heart-locked averages).
