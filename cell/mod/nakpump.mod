COMMENT
The sodium-potassium pump as the unequal pair 3:2. One cycle exports three Na+ and imports two K+, so it moves one
elementary charge outward. Each binding site is a two-state member (bound / free) with occupancy c/(c + K); the
cycle needs all three inner sodium sites and both outer potassium sites bound:

    ip = imax * (nai/(nai + kna))^3 * (ko/(ko + kk))^2       (outward charge current of the cycle)
    ina = +3 ip    (sodium out)
    ik  = -2 ip    (potassium in)

The half-points kna, kk are the sites' binding half-points (readings of where each site's two members are equal).
ENDCOMMENT

NEURON {
    SUFFIX nakpump
    USEION na READ nai WRITE ina
    USEION k READ ko WRITE ik
    RANGE imax, kna, kk, ip
}

UNITS {
    (mA) = (milliamp)
    (mM) = (milli/liter)
}

PARAMETER {
    imax = 0 (mA/cm2)
    kna = 10 (mM)
    kk = 1.5 (mM)
}

ASSIGNED {
    nai (mM) ko (mM)
    ina (mA/cm2) ik (mA/cm2) ip (mA/cm2)
}

BREAKPOINT {
    ip = imax * (nai / (nai + kna))^3 * (ko / (ko + kk))^2
    ina = 3 * ip
    ik = -2 * ip
}
