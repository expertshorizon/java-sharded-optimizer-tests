package com.trainingsim.optimizer;

import com.trainingsim.Rational;

public final class GradientClipper {

    private GradientClipper() {
    }

    public static ClipResult clip(Rational[] fullGradient, Rational clipThreshold) {
        Rational l1Norm = Rational.ZERO;
        for (Rational component : fullGradient) {
            l1Norm = l1Norm.add(component.abs());
        }
        Rational scale = Rational.ONE;
        if (l1Norm.compareTo(clipThreshold) > 0) {
            scale = clipThreshold.div(l1Norm);
        }
        Rational[] scaled = new Rational[fullGradient.length];
        for (int i = 0; i < fullGradient.length; i++) {
            scaled[i] = fullGradient[i].mul(scale);
        }
        return new ClipResult(l1Norm, scale, scaled);
    }
}
