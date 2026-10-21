package com.trainingsim;

import com.trainingsim.io.EventLogReader;
import com.trainingsim.io.OutputWriter;
import com.trainingsim.model.Config;
import com.trainingsim.model.StepTrace;
import com.trainingsim.model.SubmitEvent;
import com.trainingsim.model.TrainingState;
import com.trainingsim.optimizer.CheckpointManager;
import com.trainingsim.optimizer.StepResolver;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class Main {

    private Main() {
    }

    public static void main(String[] args) throws Exception {
        Map<String, String> options = parseArgs(args);

        Config config = EventLogReader.readConfig(Path.of(options.get("--config")));
        List<SubmitEvent> events = EventLogReader.readEvents(Path.of(options.get("--log")));

        TrainingState state;
        if (options.containsKey("--checkpoint-in")) {
            Map<String, Object> checkpointJson = EventLogReader.readCheckpoint(Path.of(options.get("--checkpoint-in")));
            state = CheckpointManager.load(checkpointJson, config);
        } else {
            state = new TrainingState(config.paramSize);
        }

        List<StepTrace> trace = new ArrayList<>();
        for (SubmitEvent event : events) {
            trace.addAll(StepResolver.ingest(event, config, state));
        }

        OutputWriter.writeResult(Path.of(options.get("--out")), config, state, trace);

        if (options.containsKey("--checkpoint-out")) {
            Map<String, Object> checkpointJson = CheckpointManager.save(state);
            OutputWriter.writeCheckpoint(Path.of(options.get("--checkpoint-out")), checkpointJson);
        }
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> options = new HashMap<>();
        for (int i = 0; i < args.length; i += 2) {
            options.put(args[i], args[i + 1]);
        }
        return options;
    }
}
