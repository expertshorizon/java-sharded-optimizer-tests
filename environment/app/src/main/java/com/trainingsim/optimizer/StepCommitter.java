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
        for (SubmitEvent event : chosen) {
            workers.add(event.workerId);
        }
        boolean poisoned = chosen.get(0).poisoned;
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
                Rational component = event.gradient != null ? event.gradient.get(i) : Rational.ZERO;
                aggregated[i] = aggregated[i].add(component);
            }
        }
        Rational quorum = Rational.ofLong(config.quorumSize);
        for (int i = 0; i < config.paramSize; i++) {
            aggregated[i] = aggregated[i].div(quorum);
        }

        int shardSize = config.shardSize();
        Rational[] clipped = new Rational[config.paramSize];
        Rational reportedL1 = null;
        Rational reportedScale = null;
        for (int shardId = 0; shardId < config.numShards; shardId++) {
            int start = shardId * shardSize;
            int end = start + shardSize;
            Rational[] slice = new Rational[shardSize];
            for (int i = start; i < end; i++) {
                slice[i - start] = aggregated[i];
            }
            ClipResult shardClip = GradientClipper.clip(slice, config.clipThreshold);
            for (int i = start; i < end; i++) {
                clipped[i] = shardClip.gradient[i - start];
            }
            if (shardId == 0) {
                reportedL1 = shardClip.l1Norm;
                reportedScale = shardClip.scale;
            }
        }

        for (int i = 0; i < config.paramSize; i++) {
            Rational retained = config.momentumBeta.mul(state.momentum[i]);
            Rational incoming = Rational.ONE.sub(config.momentumBeta).mul(clipped[i]);
            Rational newMomentum = retained.add(incoming);
            state.momentum[i] = newMomentum;
            state.params[i] = state.params[i].sub(config.learningRate.mul(newMomentum));
        }

        return new StepTrace(step, "committed", workers, reportedL1, reportedScale, rejections);
    }
}
