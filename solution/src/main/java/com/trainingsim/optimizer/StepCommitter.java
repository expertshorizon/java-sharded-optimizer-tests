package com.trainingsim.optimizer;

import com.trainingsim.Rational;
import com.trainingsim.model.Config;
import com.trainingsim.model.Rejection;
import com.trainingsim.model.StepTrace;
import com.trainingsim.model.SubmitEvent;
import com.trainingsim.model.TrainingState;

import java.util.ArrayList;
import java.util.List;

public final class StepCommitter {

    private StepCommitter() {
    }

    public static StepTrace commit(int step, List<SubmitEvent> chosen, Config config, TrainingState state) {
        List<Integer> workers = new ArrayList<>();
        boolean poisoned = false;
        for (SubmitEvent event : chosen) {
            workers.add(event.workerId);
            if (event.poisoned) {
                poisoned = true;
            }
        }
        List<Rejection> rejections = state.rejectionsByStep.getOrDefault(step, new ArrayList<>());

        if (poisoned) {
            return new StepTrace(step, "poisoned_skip", workers, null, null, rejections);
        }

        Rational[] aggregated = new Rational[config.paramSize];
        for (int i = 0; i < config.paramSize; i++) {
            aggregated[i] = Rational.ZERO;
        }
        for (SubmitEvent event : chosen) {
            for (int i = 0; i < config.paramSize; i++) {
                aggregated[i] = aggregated[i].add(event.gradient.get(i));
            }
        }
        Rational quorum = Rational.ofLong(config.quorumSize);
        for (int i = 0; i < config.paramSize; i++) {
            aggregated[i] = aggregated[i].div(quorum);
        }

        ClipResult clip = GradientClipper.clip(aggregated, config.clipThreshold);

        for (int i = 0; i < config.paramSize; i++) {
            Rational retained = config.momentumBeta.mul(state.momentum[i]);
            Rational incoming = Rational.ONE.sub(config.momentumBeta).mul(clip.gradient[i]);
            Rational newMomentum = retained.add(incoming);
            state.momentum[i] = newMomentum;
            state.params[i] = state.params[i].sub(config.learningRate.mul(newMomentum));
        }

        return new StepTrace(step, "committed", workers, clip.l1Norm, clip.scale, rejections);
    }
}
