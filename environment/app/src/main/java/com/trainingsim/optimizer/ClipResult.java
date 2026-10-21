package com.trainingsim.optimizer;

import com.trainingsim.Rational;

public final class ClipResult {

    public final Rational l1Norm;
    public final Rational scale;
    public final Rational[] gradient;

    public ClipResult(Rational l1Norm, Rational scale, Rational[] gradient) {
        this.l1Norm = l1Norm;
        this.scale = scale;
        this.gradient = gradient;
    }
}
