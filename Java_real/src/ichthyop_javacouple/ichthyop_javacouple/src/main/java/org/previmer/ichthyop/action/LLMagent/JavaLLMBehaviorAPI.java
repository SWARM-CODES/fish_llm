package org.previmer.ichthyop.action.LLMagent;

import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.models.ChatModel;
import com.openai.models.responses.Response;
import com.openai.models.responses.ResponseCreateParams;
import com.openai.models.chat.completions.ChatCompletion;
import com.openai.models.chat.completions.ChatCompletionCreateParams;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class JavaLLMBehaviorAPI {
    private final String model;
    private final OpenAIClient client;
    private final String promptTemplate;

    public JavaLLMBehaviorAPI(String configPath, String promptPath) throws Exception {
        // Load config
        JSONObject config = new JSONObject(Files.readString(Paths.get(configPath)));
        String apiKey = config.getString("api_key");
        String endpoint = config.getString("azure_endpoint");
        String apiVersion = config.getString("api_version");
        String model = config.getString("model");
        this.model = model;  
        this.client = OpenAIOkHttpClient.builder()
                .apiKey(apiKey)
                .baseUrl(endpoint + "/openai/deployments/" + model)
                .queryParams(Map.of("api-version", List.of(apiVersion)))
                .build();

        // Initialize client
        //this.client = OpenAIOkHttpClient.builder()
        //        .apiKey(apiKey)
        //        .build();

        // Load prompt template
        this.promptTemplate = Files.readString(Paths.get(promptPath));
    }

    public double[] getBehavior(JSONObject particleState, JSONArray historyState) throws Exception {
        String prompt = buildPrompt(particleState, historyState);
        int retries = 0;
        int maxretries = 3;
        double[] defaultMovement = new double[]{0.0, 0.0, 0.0};

        while (retries < maxretries) {
          try {
              ChatCompletionCreateParams params = ChatCompletionCreateParams.builder()
                    .model(ChatModel.of(model))  // e.g. "gpt-4-0125-preview"
                    .addUserMessage(prompt)
                    .maxTokens(400)
                    .temperature(0.7)
                    .build();

        
              ChatCompletion completion = client.chat().completions().create(params);
              String reply = completion.choices().get(0).message().content().orElse("").trim();
              //System.out.println("LLM raw reply: " + reply);
              return parseMovement(reply);
          } catch (Exception e) {
              retries++;
            //System.err.println("Error in LLM call (attempt " + retries + "/" + maxRetries + "): " + e.getMessage());
              try {
                    Thread.sleep((long) Math.pow(2, retries) * 5000);  // Exponential backoff
              } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();  // Good practice
              }
          }
          
          
        }
        return defaultMovement;
    }

    private String buildPrompt(JSONObject particle, JSONArray history) {
        StringBuilder dynamic = new StringBuilder();
        for (String key : particle.keySet()) {
            Object val = particle.get(key);
            switch (key) {
                case "x" -> dynamic.append("- Position:\n    - Longitude: ").append(val).append(" °\n");
                case "y" -> dynamic.append("    - Latitude: ").append(val).append(" °\n");
                case "z" -> dynamic.append("    - Z (Depth, m): ").append(val).append(" m\n");
                case "u" -> dynamic.append("- Flow Velocity:\n    - U (East-West speed, m/s): ").append(val).append(" m/s\n");
                case "v" -> dynamic.append("    - V (North-South speed, m/s): ").append(val).append(" m/s\n");
                case "w" -> dynamic.append("    - W (Vertical speed, m/s): ").append(val).append(" m/s\n");
                case "temperature" -> dynamic.append("- Current Temperature: ").append(val).append(" °C\n");
                case "bathymetry" -> dynamic.append("- Bathymetry Depth: ").append(val).append(" m\n");
                case "day" -> dynamic.append("- Seconds: ").append(val).append("\n");
                default -> dynamic.append("- ").append(key).append(": ").append(val).append("\n");
            }
        }

        StringBuilder historyStr = new StringBuilder();
        for (int i = 0; i < history.length(); i++) {
            JSONObject entry = history.getJSONObject(i);
            historyStr.append("Iteration ").append(entry.optInt("ite", i)).append(", Step ").append(i + 1).append(": ");
            List<String> kvPairs = new ArrayList<>();
            for (String key : entry.keySet()) {
                if (!key.equals("ite")) {
                    Object val = entry.get(key);
                    kvPairs.add(key + "=" + val);
                }
            }
            historyStr.append(String.join(", ", kvPairs)).append("\n");
        }

        return promptTemplate
                .replace("{dynamic_particle_state}", dynamic.toString().trim())
                .replace("{history_str}", historyStr.toString().trim());
    }

    private double[] parseMovement(String response) {
        if (!response.contains("Movement Vector:"))
            throw new RuntimeException("Malformed response: " + response);
        String line = response.split("Movement Vector:")[1].trim();
        String[] parts = line.replaceAll("[^0-9eE.,\\-+]", "").split(",");
        return new double[] {
                Double.parseDouble(parts[0].trim()),
                Double.parseDouble(parts[1].trim()),
                Double.parseDouble(parts[2].trim())
        };
    }
}

