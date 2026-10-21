package com.trainingsim.optimizer;

import com.trainingsim.model.Config;
import com.trainingsim.model.Rejection;
import com.trainingsim.model.StepTrace;
import com.trainingsim.model.SubmitEvent;
import com.trainingsim.model.TrainingState;

import java.util.ArrayList;
import java.util.List;

public final class StepResolver {

    private StepResolver() {
    }

    public static List<StepTrace> drain(Config config, TrainingState state) {
        List<StepTrace> committed = new ArrayList<>();
        cascade(config, state, committed);
        return committed;
    }

    public static List<StepTrace> ingest(SubmitEvent event, Config config, TrainingState state) {
        List<StepTrace> committed = new ArrayList<>();
        AdmissionDecision decision = StalenessGuard.classify(state.committedStep, event.step, config.maxStaleness);

        if (decision == AdmissionDecision.ACCEPT) {
            List<SubmitEvent> bucket = state.pending.computeIfAbsent(event.step, key -> new ArrayList<>());
            boolean duplicate = false;
            for (SubmitEvent existing : bucket) {
                if (existing.workerId == event.workerId) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                decision = AdmissionDecision.REJECT_DUPLICATE;
            } else if (bucket.size() >= config.quorumSize) {
                decision = AdmissionDecision.REJECT_SUPERFLUOUS;
            } else {
                bucket.add(event);
            }
        }

        if (decision != AdmissionDecision.ACCEPT) {
            Rejection rejection = new Rejection(event.workerId, event.step, reasonFor(decision));
            state.rejectionsFor(event.step).add(rejection);
            state.allRejections.add(rejection);
        }

        cascade(config, state, committed);
        return committed;
    }

    private static void cascade(Config config, TrainingState state, List<StepTrace> committed) {
        while (true) {
            int frontier = state.frontier();
            List<SubmitEvent> bucket = state.pending.get(frontier);
            if (bucket == null || bucket.size() < config.quorumSize) {
                break;
            }
            StepTrace trace = StepCommitter.commit(frontier, bucket, config, state);
            committed.add(trace);
            state.pending.remove(frontier);
            state.rejectionsByStep.remove(frontier);
            state.committedStep = frontier;
        }
    }

    private static String reasonFor(AdmissionDecision decision) {
        switch (decision) {
            case REJECT_LATE:
                return "late";
            case REJECT_STALE:
                return "stale";
            case REJECT_DUPLICATE:
                return "duplicate";
            case REJECT_SUPERFLUOUS:
                return "superfluous";
            default:
                throw new IllegalStateException("not a rejection: " + decision);
        }
    }
}
