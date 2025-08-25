package org.previmer.ichthyop.action.LLMagent;

import java.sql.*;
import java.io.File;
import java.util.Optional;

public class ParticleHistoryDB {

    private static final String DB_DIR = "./DB_tempdir";
    private static final String DB_FILE = DB_DIR + "/particle_history.db";
    private static final String DB_URL = "jdbc:sqlite:" + DB_FILE;

    public ParticleHistoryDB() {
        initDB();
    }

    private void initDB() {
        File dir = new File(DB_DIR);
        if (!dir.exists()) dir.mkdirs();

        try (Connection conn = DriverManager.getConnection(DB_URL)) {
            String sql = """
                CREATE TABLE IF NOT EXISTS particle_history (
                    particle_id INTEGER PRIMARY KEY,
                    last_llm_update REAL,
                    last_history_update REAL,
                    history TEXT
                );
            """;
            try (PreparedStatement stmt = conn.prepareStatement(sql)) {
                stmt.execute();
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public Optional<String> loadHistoryState(int particleId) {
        try (Connection conn = DriverManager.getConnection(DB_URL)) {
            String sql = "SELECT history FROM particle_history WHERE particle_id = ?";
            try (PreparedStatement stmt = conn.prepareStatement(sql)) {
                stmt.setInt(1, particleId);
                ResultSet rs = stmt.executeQuery();
                if (rs.next()) {
                    return Optional.ofNullable(rs.getString("history"));
                }
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return Optional.empty();
    }

    public void saveHistoryState(int particleId, String historyJson, double time,
                                 boolean updateHistory, boolean updateLLM) {

        try (Connection conn = DriverManager.getConnection(DB_URL)) {
            String checkSql = "SELECT last_llm_update, last_history_update FROM particle_history WHERE particle_id=?";
            try (PreparedStatement checkStmt = conn.prepareStatement(checkSql)) {
                checkStmt.setInt(1, particleId);
                ResultSet rs = checkStmt.executeQuery();

                boolean exists = rs.next();
                Double lastLLM = exists ? rs.getDouble("last_llm_update") : null;
                Double lastHist = exists ? rs.getDouble("last_history_update") : null;

                double llmTime = updateLLM ? time : (lastLLM != null ? lastLLM : time);
                double histTime = updateHistory ? time : (lastHist != null ? lastHist : time);

                String sql = exists
                        ? "UPDATE particle_history SET history=?, last_llm_update=?, last_history_update=? WHERE particle_id=?"
                        : "INSERT INTO particle_history (history, last_llm_update, last_history_update, particle_id) VALUES (?, ?, ?, ?)";

                try (PreparedStatement stmt = conn.prepareStatement(sql)) {
                    stmt.setString(1, historyJson);
                    stmt.setDouble(2, llmTime);
                    stmt.setDouble(3, histTime);
                    stmt.setInt(4, particleId);
                    stmt.executeUpdate();
                }
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public double[] getLastUpdateTimes(int particleId) {
        try (Connection conn = DriverManager.getConnection(DB_URL)) {
            String sql = "SELECT last_llm_update, last_history_update FROM particle_history WHERE particle_id=?";
            try (PreparedStatement stmt = conn.prepareStatement(sql)) {
                stmt.setInt(1, particleId);
                ResultSet rs = stmt.executeQuery();
                if (rs.next()) {
                    return new double[]{rs.getDouble("last_llm_update"), rs.getDouble("last_history_update")};
                }
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return new double[]{Double.NaN, Double.NaN};
    }
}

