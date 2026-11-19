COMMENT
The resting pair: a sodium leak and a potassium leak as two separate members, each with its own reversal (read from
the concentrations). Their sum is the single leak of a cell that does not track ions; kept apart, they are the two
members the pump restores at 3:2.
ENDCOMMENT

NEURON {
    SUFFIX nakleak
    USEION na READ ena WRITE ina
    USEION k READ ek WRITE ik
    RANGE gnal, gkl, inal, ikl
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gnal = 0 (S/cm2)
    gkl = 0 (S/cm2)
}

ASSIGNED {
    v (mV) ena (mV) ek (mV)
    ina (mA/cm2) ik (mA/cm2) inal (mA/cm2) ikl (mA/cm2)
}

BREAKPOINT {
    inal = gnal * (v - ena)
    ikl = gkl * (v - ek)
    ina = inal
    ik = ikl
}
