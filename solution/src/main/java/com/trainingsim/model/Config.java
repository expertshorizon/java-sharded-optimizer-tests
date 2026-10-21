package com.trainingsim.model;

import com.trainingsim.Rational;

import java.util.Map;

public final class Config {

    public final int numWorkers;
    public final int paramSize;
    public final int numShards;
    public final int quorumSize;
    public final int maxStaleness;
    public final Rational clipThreshold;
    public final Rational learningRate;
    public final Rational momentumBeta;

    public Config(
            int numWorkers,
            int paramSize,
            int numShards,
            int quorumSize,
            int maxStaleness,
            Rational clipThreshold,
            Rational learningRate,
            Rational momentumBeta) {
        this.numWorkers = numWorkers;
        this.paramSize = paramSize;
        this.numShards = numShards;
        this.quorumSize = quorumSize;
        this.maxStaleness = maxStaleness;
        this.clipThreshold = clipThreshold;
        this.learningRate = learningRate;
        this.momentumBeta = momentumBeta;
    }

    public static Config fromJson(Map<String, Object> json) {
        return new Config(
                intField(json, "num_workers"),
                intField(json, "param_size"),
                intField(json, "num_shards"),
                intField(json, "quorum_size"),
                intField(json, "max_staleness"),
                Rational.parseDecimal((String) json.get("clip_threshold")),
                Rational.parseDecimal((String) json.get("learning_rate")),
                Rational.parseDecimal((String) json.get("momentum_beta")));
    }

    private static int intField(Map<String, Object> json, String key) {
        return ((Long) json.get(key)).intValue();
    }

    public int shardSize() {
        return paramSize / numShards;
    }

    public int shardStart(int shardId) {
        int base = paramSize / numShards;
        int remainder = paramSize % numShards;
        return shardId * base + Math.min(shardId, remainder);
    }

    public int shardEnd(int shardId) {
        int base = paramSize / numShards;
        int remainder = paramSize % numShards;
        return shardStart(shardId) + base + (shardId < remainder ? 1 : 0);
    }
}
