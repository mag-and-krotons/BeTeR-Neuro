COMMENT
Well-mixed accumulation of intracellular sodium and potassium in a cylinder of diameter diam: every charge that
crosses the membrane changes the concentration inside by flux * area / volume = flux * 4 / diam.
Outside concentrations are held (a large bath).
ENDCOMMENT

NEURON {
    SUFFIX ionacc
    USEION na READ ina WRITE nai
    USEION k READ ik WRITE ki
    RANGE on
}

UNITS {
    (mA) = (milliamp)
    (mM) = (milli/liter)
    (um) = (micron)
    FARADAY = (faraday) (coulomb)
}

PARAMETER { on = 1 }

ASSIGNED { ina (mA/cm2) ik (mA/cm2) diam (um) }

STATE { nai (mM) ki (mM) }

INITIAL { }

BREAKPOINT { SOLVE conc METHOD cnexp }

DERIVATIVE conc {
    nai' = -on * ina * 4 * (1e4) / (diam * FARADAY)
    ki' = -on * ik * 4 * (1e4) / (diam * FARADAY)
}
