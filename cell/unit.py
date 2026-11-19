"""The network unit, derived from the live cell.

Author of the programme: Abhijit Singh, 19 November 2026.

In the live cell (cell.py, ions="live") the leak's two members and the pump close the rest at -68.45 mV. The inside
sodium of this cell (80 um, well mixed) returns with a time constant of about 50 minutes (exchange.py: derived
2908 s, read 3106 s) and one spike moves the pump by 0.007 pA, so over the seconds of the body's carriers the pump is
a constant outward current I_p. A constant I_p added to a leak g_L (V - E_L) is the same leak with
E_L' = E_L - I_p / g_L. So the network unit is NEST's hh_cond_exp_traub with E_L' and V_m = V0: nothing else
changes. check_unit.py runs the live NEURON cell and this NEST unit on the same input.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell import NEST_UNIT, close_rest  # noqa: E402


def unit_params() -> dict:
    r = close_rest()
    return {"E_L": r["E_L_effective"], "V_m": r["V0"], "g_L": NEST_UNIT["g_L"], "C_m": NEST_UNIT["C_m"],
            "g_Na": NEST_UNIT["g_Na"], "g_K": NEST_UNIT["g_K"], "E_Na": NEST_UNIT["E_Na"], "E_K": NEST_UNIT["E_K"],
            "V_T": NEST_UNIT["V_T"], "E_ex": NEST_UNIT["E_ex"], "E_in": NEST_UNIT["E_in"],
            "tau_syn_ex": NEST_UNIT["tau_syn_ex"], "tau_syn_in": NEST_UNIT["tau_syn_in"], "t_ref": NEST_UNIT["t_ref"]}


if __name__ == "__main__":
    import json
    print(json.dumps(unit_params(), indent=1))
