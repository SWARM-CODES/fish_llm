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

    public double[] computeMovement(int particleId, double x, double y, double z,
                                    double u, double v, double w,
                                    double temperature, double salinity,
                                    double bathymetry, double time,
                                    boolean updateLLM, boolean updateHistory) {
        // time = age, no need to know the absolute time
        //boolean updateLLM, updateHistory;
        //boolean wasLLMCalled = false;
        //boolean wasHistoryUpdated = false;
        //System.out.println("computeMovement1() called for particle ID: " + particleId +
        //    " | updateLLM=" + updateLLM + " | updateHistory=" + updateHistory);


        double dx = 0.0, dy = 0.0, dz = 0.0;
        // Load particle history
        List<JSONObject> historyState = new ArrayList<>();
        Optional<String> historyJsonOpt = db.loadHistoryState(particleId);
        historyJsonOpt.ifPresent(json -> {
            JSONArray array = new JSONArray(json);
            for (int i = 0; i < array.length(); i++) {
                historyState.add(array.getJSONObject(i));
            }
        });

       // double[] updateTimes = db.getLastUpdateTimes(particleId);
       // double lastLLMUpdate = updateTimes[0];
       // double lastHistoryUpdate = updateTimes[1];

        //System.out.printf(
        //   "[Particle %d] age=%.2f | lastLLMUpdate=%.2f | lastHistoryUpdate=%.2f | updateLLM=%b | updateHistory=%b%n",
        //   particleId, time, lastLLMUpdate, lastHistoryUpdate, updateLLM, updateHistory
        //);
        //System.out.println("computeMovement2() called for particle ID: " + particleId +
        //    " | updateLLM=" + updateLLM + " | updateHistory=" + updateHistory);


        JSONObject currentState = new JSONObject();
        currentState.put("ite", 1);
        currentState.put("x", safeJSONValue(x)); 
        currentState.put("y", safeJSONValue(y));
        currentState.put("z", safeJSONValue(z));
        currentState.put("u", safeJSONValue(u)); 
        currentState.put("v", safeJSONValue(v)); 
        currentState.put("w", safeJSONValue(w));
        currentState.put("temperature", safeJSONValue(temperature));
        currentState.put("salinity", safeJSONValue(salinity));
        currentState.put("bathymetry", safeJSONValue(bathymetry));
        currentState.put("seconds", safeJSONValue(time));  // day is interpreted as normalized time
        //currentState.put("time", time);

        //System.out.println("computeMovement3() called for particle ID: " + particleId +
        //                   " | u=" + u + " | v=" + v + "| w=" +w);
        int movementSourceFlag = -1;  // default: failed
        try {
            if (updateLLM) {
                //wasLLMCalled = true;
                JSONArray historyArray = new JSONArray(historyState);
                //System.out.println("History array for particle ID " + particleId + ": " + historyArray.toString());
                double[] movement = llm.getBehavior(currentState, historyArray);
                //System.out.println("History array for particle ID " + particleId + ": " + java.util.Arrays.toString(movement));
                dx = movement[0];
                dy = movement[1];
                dz = movement[2];
                currentState.put("last_behavior", new JSONArray(movement));
                movementSourceFlag = 1;  // successful LLM call
               // System.out.println("LLM behavior for particle ID " + particleId + ": movement = "
               //         + java.util.Arrays.toString(movement));
            }
            // reuse last behavior
            if (!updateLLM && updateHistory) {
            JSONObject lastState = historyState.get(historyState.size() - 1);
            JSONArray lastBehavior = lastState.getJSONArray("last_behavior");
            dx = lastBehavior.getDouble(0);
            dy = lastBehavior.getDouble(1);
            dz = lastBehavior.getDouble(2);
            currentState.put("last_behavior", lastBehavior);
            movementSourceFlag = 0;  // reused previous behavior
            //System.out.println("Reused last behavior for particle ID " + particleId);
            }
            //JSONArray updatedHistory = new JSONArray(historyState);
            //db.saveHistoryState(particleId, updatedHistory.toString(), time, updateHistory, updateLLM);

        } catch (Exception e) {
            System.out.println("Caught exception while processing particle ID " + particleId);
            e.printStackTrace();
            System.err.println("Failed to compute movement — fallback to (0,0,0)");
            dx = dy = dz = 0.0;
            movementSourceFlag = -1;  // fallback due to error
        }

        if (updateHistory) {
            try{
               // wasHistoryUpdated = true;
                currentState.put("movement_source", movementSourceFlag);
                historyState.add(currentState);
                //System.out.println("Appended current state to history for particle ID " + particleId);
                JSONArray updatedHistory = new JSONArray(historyState);
                //System.out.println("Converted historyState to JSONArray for particle ID " + particleId);
                db.saveHistoryState(particleId, updatedHistory.toString(), time, updateHistory, updateLLM);
                //System.out.println("Saved history for particle ID " + particleId + " at time " + time);
            } catch (Exception e) {
                System.out.println("Failed to update history for particle ID " + particleId + " at time " + time);
                e.printStackTrace();
            }
        }

        //if (!wasLLMCalled && !wasHistoryUpdated)
            //System.out.printf("No LLM called, no memory updated. Runtime: %.3f s%n", elapsed);
        //else if (wasHistoryUpdated && !wasLLMCalled)
        //    System.out.printf("No LLM called, memory updated. Runtime: %.3f s%n", elapsed);
        //else if (wasLLMCalled && !wasHistoryUpdated)
        //    System.out.printf("LLM called, no memory update. Runtime: %.3f s%n", elapsed);
        //else
        //System.out.printf("LLM called and memory updated. Runtime");

        return new double[]{dx, dy, dz};
    }
}

