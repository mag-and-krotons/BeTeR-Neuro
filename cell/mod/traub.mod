COMMENT
Traub-Miles sodium (m^3 h) and potassium (n^4) conductances, written to be identical to NEST's
hh_cond_exp_traub (rate functions copied from models/hh_cond_exp_traub.cpp, NEST 3.x), so that the NEURON cell and
the NEST unit are one cell. V is measured from the offset vt. No temperature factor (as in NEST).
Sodium enters through m^3 h (three activating members, one inactivating); potassium leaves through n^4.
The currents are kept per mechanism (inat, ikt) so the charges of each member can be integrated separately.
ENDCOMMENT

NEURON {
    SUFFIX traub
    USEION na READ ena WRITE ina
    USEION k READ ek WRITE ik
    RANGE gnabar, gkbar, vt, inat, ikt
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gnabar = 0.1 (S/cm2)
    gkbar = 0.03 (S/cm2)
    vt = -63 (mV)
}

ASSIGNED {
    v (mV)
    ena (mV)
    ek (mV)
    ina (mA/cm2)
    ik (mA/cm2)
    inat (mA/cm2)
    ikt (mA/cm2)
    am (/ms) bm (/ms) ah (/ms) bh (/ms) an (/ms) bn (/ms)
}

STATE { m h n }

BREAKPOINT {
    SOLVE states METHOD cnexp
    inat = gnabar * m * m * m * h * (v - ena)
    ikt = gkbar * n * n * n * n * (v - ek)
    ina = inat
    ik = ikt
}

INITIAL {
    rates(v)
    m = am / (am + bm)
    h = ah / (ah + bh)
    n = an / (an + bn)
}

DERIVATIVE states {
    rates(v)
    m' = am - (am + bm) * m
    h' = ah - (ah + bh) * h
    n' = an - (an + bn) * n
}

FUNCTION xexp(x, y) {
    : x / (exp(x/y) - 1), continuous through x = 0
    if (fabs(x / y) < 1e-6) {
        xexp = y * (1 - x / y / 2)
    } else {
        xexp = x / (exp(x / y) - 1)
    }
}

PROCEDURE rates(vm (mV)) {
    LOCAL u
    UNITSOFF
    u = vm - vt
    an = 0.032 * xexp(15 - u, 5)
    bn = 0.5 * exp((10 - u) / 40)
    am = 0.32 * xexp(13 - u, 4)
    bm = 0.28 * xexp(u - 40, 5)
    ah = 0.128 * exp((17 - u) / 18)
    bh = 4 / (1 + exp((40 - u) / 5))
    UNITSON
}
