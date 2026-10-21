package com.trainingsim.model;

import com.trainingsim.Rational;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public final class SubmitEvent {

    public final int workerId;
    public final int step;
    public final boolean poisoned;
    public final List<Rational> gradient;

    public SubmitEvent(int workerId, int step, boolean poisoned, List<Rational> gradient) {
        this.workerId = workerId;
        this.step = step;
        this.poisoned = poisoned;
        this.gradient = gradient;
    }

    @SuppressWarnings("unchecked")
    public static SubmitEvent fromJson(Map<String, Object> json) {
        int workerId = ((Long) json.get("worker_id")).intValue();
        int step = ((Long) json.get("step")).intValue();
        boolean poisoned = Boolean.TRUE.equals(json.get("poisoned"));
        List<Rational> gradient = null;
        if (!poisoned) {
            List<Object> raw = (List<Object>) json.get("gradient");
            gradient = new ArrayList<>(raw.size());
            for (Object component : raw) {
                gradient.add(Rational.parseDecimal((String) component));
            }
        }
        return new SubmitEvent(workerId, step, poisoned, gradient);
    }
}
