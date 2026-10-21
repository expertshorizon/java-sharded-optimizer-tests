package com.trainingsim.model;

import com.trainingsim.Rational;

import java.util.LinkedHashMap;
import java.util.Map;

public final class TrainingState {

    public final Rational[] params;
    public final Rational[] momentum;
    public int committedStep;
    public final Map<Integer, java.util.List<SubmitEvent>> pending;
    public final Map<Integer, java.util.List<Rejection>> rejectionsByStep;
    public final java.util.List<Rejection> allRejections;

    public TrainingState(int paramSize) {
        this.params = new Rational[paramSize];
        this.momentum = new Rational[paramSize];
        for (int i = 0; i < paramSize; i++) {
            this.params[i] = Rational.ZERO;
            this.momentum[i] = Rational.ZERO;
        }
        this.committedStep = -1;
        this.pending = new LinkedHashMap<>();
        this.rejectionsByStep = new LinkedHashMap<>();
        this.allRejections = new java.util.ArrayList<>();
    }

    public int frontier() {
        return committedStep + 1;
    }

    public java.util.List<Rejection> rejectionsFor(int step) {
        return rejectionsByStep.computeIfAbsent(step, k -> new java.util.ArrayList<>());
    }
}
