package org.previmer.ichthyop.action.LLMagent;

import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.models.ChatModel;
import com.openai.models.chat.completions.ChatCompletion;
import com.openai.models.chat.completions.ChatCompletionCreateParams;
import org.json.JSONArray;
import org.json.JSONObject;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Batched LLM behavior (OpenAI Java SDK 4.x, direct OpenAI):
 * - If particleId == 1:
 *   - load ALL particles from DB
 *   - make ONE chat call (one system + many user messages)
 *   - parse ONE JSON array [{particle, dx, dy, dz}, ...]
 *   - write each vector back to DB as last_behavior on the latest entry
 * - Then (for any particleId) return that id's latest [dx,dy,dz]
 */
public class JavaLLMBehaviorAPI {

    private final String model;
    private final OpenAIClient client;
    private final String systemPrompt;
    private final ParticleHistoryDB db;

    public JavaLLMBehaviorAPI(String configPath, String promptPath) throws Exception {
        // Load config
        JSONObject config = new JSONObject(Files.readString(Paths.get(configPath)));
        String apiKey = config.getString("api_key");
        String mdl    = config.getString("model"); // e.g., "gpt-4o-mini"
        this.model = mdl;

        // OpenAI direct (no custom base URL required)
        this.client = OpenAIOkHttpClient.builder()
                .apiKey(apiKey)
                .build();
        // (Alternatively: OpenAIOkHttpClient.fromEnv() and set OPENAI_API_KEY in env.)

        // Single system prompt
        this.systemPrompt = Files.readString(Paths.get(promptPath));

        // DB handle
        this.db = new ParticleHistoryDB();
    }

    /** Entry point used by ParticleLLMAgent. */
    public double[] getBehavior(int particleId) {
        try {
            if (particleId == 1) {
                batchUpdateAllParticles();
            }
            double[] mv = readLatestBehaviorFromDB(particleId);
            return (mv != null) ? mv : new double[]{0.0, 0.0, 0.0};
        } catch (Exception e) {
            System.err.println("[LLM] getBehavior failed for particle " + particleId + ": " + e.getMessage());
            e.printStackTrace();
            return new double[]{0.0, 0.0, 0.0};
        }
    }

    // ------------------------------ Batch update ------------------------------

    private void batchUpdateAllParticles() throws Exception {
        // A) enumerate particle IDs present in DB
        List<Integer> ids = db.listAllParticleIds();
        if (ids == null || ids.isEmpty()) {
            System.err.println("[LLM] No particles in DB; skipping batch.");
            return;
        }

        // B) load full history + last state for each
        List<JSONArray> histories   = new ArrayList<>(ids.size());
        List<JSONObject> lastStates = new ArrayList<>(ids.size());
        for (int pid : ids) {
            JSONArray hist = loadHistoryArray(pid);
            histories.add(hist);
            lastStates.add(hist.length() > 0 ? hist.getJSONObject(hist.length() - 1) : new JSONObject());
        }

        // C) build one system + many user messages (one per particle, in the same order as ids)
        List<String> userMsgs = new ArrayList<>(ids.size());
        for (int i = 0; i < ids.size(); i++) {
            int pid = ids.get(i);
            userMsgs.add(buildUserMessage(pid, lastStates.get(i), histories.get(i)));
        }

        // D) single chat call → expect ONE JSON array (same order as user messages)
        String reply = callOnce(systemPrompt, userMsgs, Math.max(6000, ids.size() * 64));

        // E) parse → Map<pid, [dx,dy,dz]>
        Map<Integer,double[]> byId = parseBatchJsonToMap(reply, ids);

        // F) write back each vector into the latest history entry for that particle
        for (int i = 0; i < ids.size(); i++) {
            int pid = ids.get(i);
            double[] mv = byId.getOrDefault(pid, new double[]{0.0, 0.0, 0.0});
            writeBackLastBehavior(pid, histories.get(i), mv);
        }
    }

    // ------------------------------ Build messages ------------------------------

    private String buildUserMessage(int particleId, JSONObject latest, JSONArray history) {
        StringBuilder sb = new StringBuilder(512);
        sb.append("Particle ").append(particleId).append(":\n");

        sb.append("STATE:\n").append(renderState(latest)).append("\n\n");

        sb.append("HISTORY:\n");
        if (history != null && history.length() > 0) {
            for (int i = 0; i < history.length(); i++) {
                JSONObject e = history.getJSONObject(i);
                int ite = e.optInt("ite", i + 1);
                sb.append("Iteration ").append(ite).append(", Step ").append(i + 1).append(": ");
                List<String> kv = new ArrayList<>();
                for (String k : e.keySet()) {
                    if ("ite".equals(k)) continue;
                    kv.add(k + "=" + e.opt(k));
                }
                sb.append(String.join(", ", kv)).append("\n");
            }
        } else {
            sb.append("None\n");
        }

        // Output contract
        sb.append("Return ONE consolidated JSON array for ALL particles, in the SAME ORDER as these messages. ")
          .append("Each item MUST be: {\"particle\": ").append(particleId)
          .append(", \"dx\": <float>, \"dy\": <float>, \"dz\": <float>}.\n")
          .append("No extra text outside the JSON array.");
        return sb.toString();
    }

    private String renderState(JSONObject s) {
        StringBuilder d = new StringBuilder();
        addKV(d, s, "x", "X (km)");
        addKV(d, s, "y", "Y (km)");
        addKV(d, s, "z", "Z (m)");
        addKV(d, s, "u", "U (m/s)");
        addKV(d, s, "v", "V (m/s)");
        addKV(d, s, "w", "W (m/s)");
        addKV(d, s, "temperature", "Temperature (°C)");
        addKV(d, s, "salinity", "Salinity (psu)");
        addKV(d, s, "bathymetry", "Bathymetry (m)");
        addKV(d, s, "seconds", "Age (s)");
        return d.toString();
    }
    private void addKV(StringBuilder d, JSONObject s, String key, String label) {
        if (s != null && s.has(key) && !s.isNull(key)) {
            d.append("- ").append(label).append(": ").append(s.opt(key)).append("\n");
        }
    }

    // ------------------------------ OpenAI call (4.x) ------------------------------

    private String callOnce(String sys, List<String> userMessages, int maxTokens) throws Exception {
        int tries = 0, maxTries = 3;
        // for debug
        //System.out.println("=== PROMPT DEBUG1 ===");
        //if (sys != null && !sys.isEmpty()) System.out.println("system: " + sys);
        //if (userMessages != null) {
        //  for (String um : userMessages) if (um != null && !um.isEmpty()) System.out.println("user: " + um);
        //}
        //System.out.println("=== PROMPT DEBUG2 ===");
        //
        while (true) {
            try {
                ChatCompletionCreateParams.Builder b = ChatCompletionCreateParams.builder()
                        .model(ChatModel.of(model))
                        .maxCompletionTokens(maxTokens)
                        .temperature(1.0);

                if (sys != null && !sys.isEmpty()) {
                    b.addSystemMessage(sys);
                }
                if (userMessages != null) {
                    for (String um : userMessages) {
                        if (um != null && !um.isEmpty()) b.addUserMessage(um);
                    }
                }

                ChatCompletionCreateParams params = b.build();

                ChatCompletion completion = client.chat().completions().create(params);

                // print full raw completion (includes finish_reason, message structure, etc.)
                System.out.println("=== RAW COMPLETION DEBUG ===");
                System.out.println(completion.toString());
                System.out.println("============================");


                String reply = completion.choices().get(0).message().content().orElse("").trim();
                if (reply.isEmpty()) throw new RuntimeException("Empty LLM reply.");
                return reply;

            } catch (Exception e) {
                tries++;
                if (tries >= maxTries) throw e;
                try { Thread.sleep((long) Math.pow(2, tries) * 1000L); }
                catch (InterruptedException ie) { Thread.currentThread().interrupt(); }
            }
        }
    }

    // ------------------------------ Parse result ------------------------------

    /** Parse model reply into {particleId -> [dx,dy,dz]}. Accepts flat or a nested fallback. */
    private Map<Integer,double[]> parseBatchJsonToMap(String text, List<Integer> idOrder) {
        String json = extractJsonArray(text);
        if (json == null) throw new RuntimeException("No JSON array found in reply:\n" + text);

        JSONArray arr = new JSONArray(json);
        Map<Integer,double[]> out = new HashMap<>();
        for (int i = 0; i < arr.length(); i++) {
            JSONObject item = arr.optJSONObject(i);
            if (item == null) continue;

            int pid = item.optInt("particle", (i < idOrder.size() ? idOrder.get(i) : -1));
            Double dx = optDouble(item, "dx");
            Double dy = optDouble(item, "dy");
            Double dz = optDouble(item, "dz");

            // robust fallback if model nests values
            if (dx == null || dy == null || dz == null) {
                JSONObject br = item.optJSONObject("brief_rationale");
                JSONObject q4 = (br != null) ? br.optJSONObject("q4") : null;
                if (q4 != null) {
                    if (dx == null) dx = optDouble(q4, "dx");
                    if (dy == null) dy = optDouble(q4, "dy");
                    if (dz == null) dz = optDouble(q4, "dz");
                }
            }

            if (dx == null) dx = 0.0;
            if (dy == null) dy = 0.0;
            if (dz == null) dz = 0.0;

            out.put(pid, new double[]{dx, dy, dz});
        }
        return out;
    }

    private String extractJsonArray(String text) {
        String t = text.trim();
        if (t.startsWith("[") && t.endsWith("]")) return t;
        Matcher m = Pattern.compile("(\\[.*?\\])", Pattern.DOTALL).matcher(t);
        return m.find() ? m.group(1) : null;
    }

    private Double optDouble(JSONObject o, String k) {
        if (o == null || !o.has(k) || o.isNull(k)) return null;
        try { return o.getNumber(k).doubleValue(); }
        catch (Exception ignore) {
            try { return Double.parseDouble(String.valueOf(o.get(k))); }
            catch (Exception e) { return null; }
        }
    }

    // ------------------------------ DB helpers ------------------------------

    private JSONArray loadHistoryArray(int particleId) {
        return db.loadHistoryState(particleId)
                 .map(JSONArray::new)
                 .orElseGet(JSONArray::new);
    }

    /** Write dx,dy,dz into the latest history entry as "last_behavior"; persist with updateLLM=true. */
    private void writeBackLastBehavior(int particleId, JSONArray history, double[] mv) {
        try {
            if (history.length() == 0) {
                JSONObject e = new JSONObject();
                e.put("ite", 1);
                e.put("last_behavior", new JSONArray(mv));
                history.put(e);
            } else {
                JSONObject last = history.getJSONObject(history.length() - 1);
                last.put("last_behavior", new JSONArray(mv));
            }
            double lastTime = 0.0;
            if (history.length() > 0) {
                JSONObject last = history.getJSONObject(history.length() - 1);
                if (last.has("seconds") && !last.isNull("seconds")) {
                    try { lastTime = last.getNumber("seconds").doubleValue(); } catch (Exception ignore) {}
                }
            }
            db.saveHistoryState(particleId, history.toString(), lastTime, /*updateHistory*/ false, /*updateLLM*/ true);
        } catch (Exception e) {
            System.err.println("[LLM] Failed to write last_behavior for particle " + particleId + ": " + e.getMessage());
        }
    }

    /** Read latest last_behavior for id; null if missing. */
    private double[] readLatestBehaviorFromDB(int particleId) {
        JSONArray hist = loadHistoryArray(particleId);
        if (hist.length() == 0) return null;
        JSONObject last = hist.getJSONObject(hist.length() - 1);
        if (!last.has("last_behavior") || last.isNull("last_behavior")) return null;
        JSONArray lb = last.getJSONArray("last_behavior");
        if (lb.length() < 3) return null;
        return new double[]{lb.optDouble(0, 0.0), lb.optDouble(1, 0.0), lb.optDouble(2, 0.0)};
    }
}

