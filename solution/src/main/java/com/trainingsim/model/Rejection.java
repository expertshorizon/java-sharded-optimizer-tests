package com.trainingsim.model;

import java.util.LinkedHashMap;
import java.util.Map;

public final class Rejection {

    public final int workerId;
    public final int step;
    public final String reason;

    public Rejection(int workerId, int step, String reason) {
        this.workerId = workerId;
        this.step = step;
        this.reason = reason;
    }

    public Map<String, Object> toJson() {
        Map<String, Object> json = new LinkedHashMap<>();
        json.put("worker_id", (long) workerId);
        json.put("step", (long) step);
        json.put("reason", reason);
        return json;
    }
}
