package com.trainingsim.io;

import com.trainingsim.Rational;
import com.trainingsim.json.Json;
import com.trainingsim.model.Config;
import com.trainingsim.model.StepTrace;
import com.trainingsim.model.SubmitEvent;
import com.trainingsim.model.TrainingState;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class OutputWriter {

    private OutputWriter() {
    }

    public static void writeResult(Path path, Config config, TrainingState state, List<StepTrace> trace)
            throws IOException {
        Map<String, Object> json = new LinkedHashMap<>();
        json.put("final_state", finalStateJson(config, state));
        List<Object> traceJson = new ArrayList<>();
        for (StepTrace entry : trace) {
            traceJson.add(entry.toJson());
        }
        json.put("trace", traceJson);
        json.put("uncommitted_pending", uncommittedJson(state));
        List<Object> rejectionsJson = new ArrayList<>();
        for (com.trainingsim.model.Rejection rejection : state.allRejections) {
            rejectionsJson.add(rejection.toJson());
        }
        json.put("rejections", rejectionsJson);
        Files.writeString(path, Json.write(json));
    }

    public static void writeCheckpoint(Path path, Map<String, Object> checkpointJson) throws IOException {
        Files.writeString(path, Json.write(checkpointJson));
    }

    private static Map<String, Object> finalStateJson(Config config, TrainingState state) {
        Map<String, Object> json = new LinkedHashMap<>();
        List<Object> shards = new ArrayList<>();
        int shardSize = config.shardSize();
        for (int shardId = 0; shardId < config.numShards; shardId++) {
            int start = shardId * shardSize;
            int end = start + shardSize;
            Map<String, Object> shard = new LinkedHashMap<>();
            shard.put("shard_id", (long) shardId);
            shard.put("params", sliceToJson(state.params, start, end));
            shard.put("momentum", sliceToJson(state.momentum, start, end));
            shards.add(shard);
        }
        json.put("shards", shards);
        return json;
    }

    private static List<Object> sliceToJson(Rational[] values, int start, int end) {
        List<Object> list = new ArrayList<>(end - start);
        for (int i = start; i < end; i++) {
            list.add(values[i].toString());
        }
        return list;
    }

    private static Map<String, Object> uncommittedJson(TrainingState state) {
        Map<String, Object> json = new LinkedHashMap<>();
        for (Map.Entry<Integer, List<SubmitEvent>> entry : state.pending.entrySet()) {
            List<Object> workers = new ArrayList<>();
            for (SubmitEvent event : entry.getValue()) {
                workers.add((long) event.workerId);
            }
            json.put(String.valueOf(entry.getKey()), workers);
        }
        return json;
    }
}
