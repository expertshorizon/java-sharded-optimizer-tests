package com.trainingsim.io;

import com.trainingsim.json.Json;
import com.trainingsim.model.Config;
import com.trainingsim.model.SubmitEvent;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public final class EventLogReader {

    private EventLogReader() {
    }

    public static Config readConfig(Path path) throws IOException {
        String text = Files.readString(path);
        return Config.fromJson(Json.parseObject(text));
    }

    @SuppressWarnings("unchecked")
    public static List<SubmitEvent> readEvents(Path path) throws IOException {
        String text = Files.readString(path);
        List<Object> raw = (List<Object>) Json.parse(text);
        List<SubmitEvent> events = new ArrayList<>(raw.size());
        for (Object item : raw) {
            events.add(SubmitEvent.fromJson((Map<String, Object>) item));
        }
        return events;
    }

    public static Map<String, Object> readCheckpoint(Path path) throws IOException {
        String text = Files.readString(path);
        return Json.parseObject(text);
    }
}
