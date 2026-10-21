package com.trainingsim.optimizer;

import com.trainingsim.Rational;
import com.trainingsim.model.Config;
import com.trainingsim.model.Rejection;
import com.trainingsim.model.SubmitEvent;
import com.trainingsim.model.TrainingState;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class CheckpointManager {

    private CheckpointManager() {
    }

    public static Map<String, Object> save(TrainingState state) {
        Map<String, Object> json = new LinkedHashMap<>();
        json.put("committed_step", (long) state.committedStep);
        json.put("params", rationalArrayToJson(state.params));
        json.put("momentum", rationalArrayToJson(state.momentum));
        json.put("pending", new LinkedHashMap<>());
        json.put("rejections", rejectionsToJson(state.rejectionsByStep));
        List<Object> allRejections = new ArrayList<>();
        for (Rejection rejection : state.allRejections) {
            allRejections.add(rejection.toJson());
        }
        json.put("all_rejections", allRejections);
        return json;
    }

    @SuppressWarnings("unchecked")
    public static TrainingState load(Map<String, Object> json, Config config) {
        TrainingState state = new TrainingState(config.paramSize);
        state.committedStep = ((Long) json.get("committed_step")).intValue();

        List<Object> params = (List<Object>) json.get("params");
        List<Object> momentum = (List<Object>) json.get("momentum");
        for (int i = 0; i < config.paramSize; i++) {
            state.params[i] = Rational.parseFraction((String) params.get(i));
            state.momentum[i] = Rational.parseFraction((String) momentum.get(i));
        }

        Map<String, Object> pending = (Map<String, Object>) json.get("pending");
        for (Map.Entry<String, Object> entry : pending.entrySet()) {
            int step = Integer.parseInt(entry.getKey());
            List<Object> bucket = (List<Object>) entry.getValue();
            List<SubmitEvent> events = new ArrayList<>();
            for (Object item : bucket) {
                Map<String, Object> eventJson = (Map<String, Object>) item;
                events.add(submitEventFromCheckpointJson(eventJson));
            }
            state.pending.put(step, events);
        }

        Map<String, Object> rejections = (Map<String, Object>) json.get("rejections");
        for (Map.Entry<String, Object> entry : rejections.entrySet()) {
            int step = Integer.parseInt(entry.getKey());
            List<Object> bucket = (List<Object>) entry.getValue();
            List<Rejection> list = new ArrayList<>();
            for (Object item : bucket) {
                Map<String, Object> rejectionJson = (Map<String, Object>) item;
                list.add(new Rejection(
                        ((Long) rejectionJson.get("worker_id")).intValue(),
                        ((Long) rejectionJson.get("step")).intValue(),
                        (String) rejectionJson.get("reason")));
            }
            state.rejectionsByStep.put(step, list);
        }

        List<Object> allRejections = (List<Object>) json.get("all_rejections");
        if (allRejections == null) {
            throw new IllegalArgumentException("checkpoint missing all_rejections");
        }

        return state;
    }

    private static List<Object> rationalArrayToJson(Rational[] values) {
        List<Object> list = new ArrayList<>(values.length);
        for (Rational value : values) {
            list.add(value.toString());
        }
        return list;
    }

    private static Map<String, Object> pendingToJson(Map<Integer, List<SubmitEvent>> pending) {
        Map<String, Object> json = new LinkedHashMap<>();
        for (Map.Entry<Integer, List<SubmitEvent>> entry : pending.entrySet()) {
            List<Object> bucket = new ArrayList<>();
            for (SubmitEvent event : entry.getValue()) {
                bucket.add(submitEventToCheckpointJson(event));
            }
            json.put(String.valueOf(entry.getKey()), bucket);
        }
        return json;
    }

    private static Map<String, Object> rejectionsToJson(Map<Integer, List<Rejection>> rejections) {
        Map<String, Object> json = new LinkedHashMap<>();
        for (Map.Entry<Integer, List<Rejection>> entry : rejections.entrySet()) {
            List<Object> bucket = new ArrayList<>();
            for (Rejection rejection : entry.getValue()) {
                bucket.add(rejection.toJson());
            }
            json.put(String.valueOf(entry.getKey()), bucket);
        }
        return json;
    }

    private static Map<String, Object> submitEventToCheckpointJson(SubmitEvent event) {
        Map<String, Object> json = new LinkedHashMap<>();
        json.put("worker_id", (long) event.workerId);
        json.put("step", (long) event.step);
        json.put("poisoned", event.poisoned);
        if (event.gradient != null) {
            json.put("gradient", rationalArrayToJson(event.gradient.toArray(new Rational[0])));
        } else {
            json.put("gradient", null);
        }
        return json;
    }

    @SuppressWarnings("unchecked")
    private static SubmitEvent submitEventFromCheckpointJson(Map<String, Object> json) {
        int workerId = ((Long) json.get("worker_id")).intValue();
        int step = ((Long) json.get("step")).intValue();
        boolean poisoned = Boolean.TRUE.equals(json.get("poisoned"));
        List<Rational> gradient = null;
        if (json.get("gradient") != null) {
            List<Object> raw = (List<Object>) json.get("gradient");
            gradient = new ArrayList<>(raw.size());
            for (Object component : raw) {
                gradient.add(Rational.parseFraction((String) component));
            }
        }
        return new SubmitEvent(workerId, step, poisoned, gradient);
    }
}
