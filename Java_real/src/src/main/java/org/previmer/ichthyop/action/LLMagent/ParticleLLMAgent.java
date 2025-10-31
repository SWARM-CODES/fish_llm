package org.previmer.ichthyop.action.LLMagent;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class ParticleLLMAgent {

    private final ParticleHistoryDB db;
    private final JavaLLMBehaviorAPI llm;

    public ParticleLLMAgent(String configPath, String promptPath) throws Exception {
        this.db = new ParticleHistoryDB();
        this.llm = new JavaLLMBehaviorAPI(configPath, promptPath);
    }

    private static Object safeJSONValue(double value) {
        return (Double.isNaN(value) || Double.isInfinite(value)) ? JSONObject.NULL : value;
    }

    /**
     * Behavior:
     * - updateHistory == true  → append current state to DB and reuse last_behavior; save with (true, false)
     * - updateHistory == false && updateLLM == true → call LLM; no DB write here
     * - otherwise → reuse last known last_behavior; no DB write
     */
    public double[] computeMovement(int particleId, double x, double y, double z,
                                    double u, double v, double w,
                                    double temperature, double salinity,
                                    double bathymetry, double time,
                                    boolean updateLLM, boolean updateHistory) {

        System.out.println(
            String.format("Time: %.2f | updateLLM: %b | updateHistory: %b",
                           time, updateLLM, updateHistory)
        );


        // Load existing history JSON array (or empty)
        JSONArray history = db.loadHistoryState(particleId)
                              .map(JSONArray::new)
                              .orElseGet(JSONArray::new);

        // Prepare current state snapshot (only appended when updateHistory == true)
        JSONObject currentState = new JSONObject();
        currentState.put("ite", history.length() + 1);
        currentState.put("x",           safeJSONValue(x));
        currentState.put("y",           safeJSONValue(y));
        currentState.put("z",           safeJSONValue(z));
        currentState.put("u",           safeJSONValue(u));
        currentState.put("v",           safeJSONValue(v));
        currentState.put("w",           safeJSONValue(w));
        currentState.put("temperature", safeJSONValue(temperature));
        currentState.put("salinity",    safeJSONValue(salinity));
        currentState.put("bathymetry",  safeJSONValue(bathymetry));
        currentState.put("seconds",     safeJSONValue(time));

        double dx = 0.0, dy = 0.0, dz = 0.0;

        try {
            if (updateHistory) {
                // HISTORY MODE: append current state, reuse last_behavior; ignore updateLLM here.
                if (history.length() > 0) {
                    JSONObject last = history.getJSONObject(history.length() - 1);
                    if (last.has("last_behavior") && !last.isNull("last_behavior")) {
                        JSONArray lb = last.getJSONArray("last_behavior");
                        dx = lb.optDouble(0, 0.0);
                        dy = lb.optDouble(1, 0.0);
                        dz = lb.optDouble(2, 0.0);
                        currentState.put("last_behavior", lb);
                        currentState.put("movement_source", 0); // reused
                    } else {
                        currentState.put("last_behavior", new JSONArray(new double[]{0.0, 0.0, 0.0}));
                        currentState.put("movement_source", -1); // none available
                    }
                } else {
                    currentState.put("last_behavior", new JSONArray(new double[]{0.0, 0.0, 0.0}));
                    currentState.put("movement_source", -1);
                }

                // Append and persist with updateHistory=true, updateLLM=false
                history.put(currentState);
                db.saveHistoryState(
                    particleId,
                    history.toString(),
                    time,
                    /*updateHistory*/ true,
                    /*updateLLM*/    false
                );
                return new double[]{dx, dy, dz};
            }

            if (updateLLM) {
                // LLM MODE: call LLM; NO DB write here (batch write happens inside llm.getBehavior when id==1)
                double[] mv = llm.getBehavior(particleId);
                if (mv == null || mv.length < 3) return new double[]{0.0, 0.0, 0.0};
                return new double[]{mv[0], mv[1], mv[2]};
            }

            // Neither flag: reuse last known behavior (no DB write)
            if (history.length() > 0) {
                JSONObject last = history.getJSONObject(history.length() - 1);
                if (last.has("last_behavior") && !last.isNull("last_behavior")) {
                    JSONArray lb = last.getJSONArray("last_behavior");
                    return new double[]{
                        lb.optDouble(0, 0.0),
                        lb.optDouble(1, 0.0),
                        lb.optDouble(2, 0.0)
                    };
                }
            }
            return new double[]{0.0, 0.0, 0.0};

        } catch (Exception e) {
            System.err.println("Particle " + particleId + " computeMovement failed: " + e.getMessage());
            e.printStackTrace();
            return new double[]{0.0, 0.0, 0.0};
        }
    }
}

