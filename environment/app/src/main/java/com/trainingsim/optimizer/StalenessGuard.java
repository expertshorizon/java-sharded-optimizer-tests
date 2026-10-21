package com.trainingsim.optimizer;

public final class StalenessGuard {

    private StalenessGuard() {
    }

    public static AdmissionDecision classify(int committedStep, int targetStep, int maxStaleness) {
        int frontier = committedStep + 1;
        if (targetStep <= committedStep) {
            return AdmissionDecision.REJECT_LATE;
        }
        int lookahead = targetStep - frontier;
        if (lookahead >= maxStaleness) {
            return AdmissionDecision.REJECT_STALE;
        }
        return AdmissionDecision.ACCEPT;
    }
}
