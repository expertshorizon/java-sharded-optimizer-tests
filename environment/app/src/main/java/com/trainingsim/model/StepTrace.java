package com.trainingsim.model;

import com.trainingsim.Rational;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class StepTrace {

    public final int step;
    public final String status;
    public final List<Integer> workers;
    public final Rational l1Norm;
    public final Rational clipScale;
    public final List<Rejection> rejections;

    public StepTrace(
            int step,
            String status,
            List<Integer> workers,
            Rational l1Norm,
            Rational clipScale,
            List<Rejection> rejections) {
        this.step = step;
        this.status = status;
        this.workers = workers;
        this.l1Norm = l1Norm;
        this.clipScale = clipScale;
        this.rejections = rejections;
    }

    public Map<String, Object> toJson() {
        Map<String, Object> json = new LinkedHashMap<>();
        json.put("step", (long) step);
        json.put("status", status);
        List<Object> workerList = new ArrayList<>();
        for (Integer worker : workers) {
            workerList.add((long) worker);
        }
        json.put("workers", workerList);
        json.put("l1_norm", l1Norm == null ? null : l1Norm.toString());
        json.put("clip_scale", clipScale == null ? null : clipScale.toString());
        List<Object> rejectionList = new ArrayList<>();
        for (Rejection rejection : rejections) {
            rejectionList.add(rejection.toJson());
        }
        json.put("rejections", rejectionList);
        return json;
    }
}
