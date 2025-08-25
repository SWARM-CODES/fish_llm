package org.previmer.ichthyop.action;

import org.previmer.ichthyop.particle.IParticle;
import org.previmer.ichthyop.action.LLMagent.ParticleLLMAgent;
import org.previmer.ichthyop.action.LLMagent.ParticleHistoryDB;
import org.previmer.ichthyop.action.LLMagent.JavaLLMBehaviorAPI;
import java.util.HashMap;
import java.util.Map;


public class LLMParticleMovementAction extends AbstractAction {

    private boolean enableAction;
    private String salinity_field;
    private String temperature_field;

    private double UPDATE_LLM_INTERVAL;
    private double UPDATE_HISTORY_INTERVAL;

    private ParticleLLMAgent agent;
    private final Map<Integer, double[]> lastMovementMap = new HashMap<>();    

    @Override
    public void loadParameters() throws Exception {
        enableAction = getParameter("enable").equals("1");
        if (enableAction) {
          salinity_field = getParameter("salinity_field");
          temperature_field = getParameter("temperature_field");

          UPDATE_LLM_INTERVAL = Double.parseDouble(getParameter("update_llm_interval"));
          UPDATE_HISTORY_INTERVAL = Double.parseDouble(getParameter("update_history_interval"));

          getSimulationManager().getDataset().requireVariable(temperature_field, getClass());
          getSimulationManager().getDataset().requireVariable(salinity_field, getClass());

          agent = new ParticleLLMAgent("config.json", "prompt.txt");
        }
     }


    @Override
    public void execute(IParticle particle) {
        if (!enableAction) {
            return;
        }
        try {
            double age = particle.getAge();
            double time = getSimulationManager().getTimeManager().getTime();
            double dt = getSimulationManager().getTimeManager().get_dt();
            
            boolean updateLLM = (age % UPDATE_LLM_INTERVAL) < 1e-3;
            boolean updateHistory = (age % UPDATE_HISTORY_INTERVAL) < 1e-3;
            
           // System.out.printf(
           //         "time=%.4f | UPDATE_LLM_INTERVAL=%.4f | UPDATE_HISTORY_INTERVAL=%.4f | updateLLM=%b | updateHistory=%b%n",
           //         time, UPDATE_LLM_INTERVAL, UPDATE_HISTORY_INTERVAL, updateLLM, updateHistory
           // );
            double dx, dy, dz;
            int particleId = particle.getIndex();



            if (!updateLLM && !updateHistory) {
              double[] cached = lastMovementMap.get(particleId);
              if (cached == null) return; // no cached result available, skip movement
              dx = cached[0];
              dy = cached[1];
              dz = cached[2];
            } else {

              double salinity = getSimulationManager().getDataset().get(salinity_field, particle.getGridCoordinates(), time).doubleValue();
              double temperature = getSimulationManager().getDataset().get(temperature_field, particle.getGridCoordinates(), time).doubleValue();
              double bathymetry = particle.getDepth();

              double lon = particle.getLon();
              double lat = particle.getLat();
              double x = particle.getX();
              double y = particle.getY();
              double z = particle.getZ();
              double u = getSimulationManager().getDataset().get_dUx(particle.getGridCoordinates(), time);
              double v = getSimulationManager().getDataset().get_dVy(particle.getGridCoordinates(), time);
              double w = getSimulationManager().getDataset().get_dWz(particle.getGridCoordinates(), time);
              //System.out.printf("BUOYANCY LLM DEBUG - Particle %d | Time: %.2f | X: %.4f | Y: %.4f%n | U: %.4f%n | V: %.4f%n | W: %.4f%n",
              //      particle.getIndex(), age, lon, lat, u, v, w);

            // Call Java LLM agent directly
            //  double[] result = agent.computeMovement(
            //        particleId, lon, lat, z, u, v, w, temperature, salinity, -bathymetry, age, updateLLM, updateHistory);

              dx = 0.0;    //result[0];
              dy = 0.0;    //result[1];
              dz = -0.001; //result[2];
              lastMovementMap.put(particleId, new double[]{dx, dy, dz});  
            } 

            // Always apply movement regardless of source
            double x = particle.getX();
            double y = particle.getY();
            double z = particle.getZ();
            double bathymetry = particle.getDepth();


            //System.out.printf("Before scaling → Particle %d: dx=%.5f, dy=%.5f, dz=%.5f%n",
            //                  particleId, dx, dy, dz);            


            int i = (int) Math.round(x);
            int j = (int) Math.round(y);
            dx = (dx * dt) / getSimulationManager().getDataset().getdxi(j, i);
            dy = (dy * dt) / getSimulationManager().getDataset().getdeta(j, i);
            dz = getSimulationManager().getDataset().depth2z(x, y, bathymetry + (dz * dt)) - z;


            //System.out.printf("Afer scaling → Particle %d: z=%.5f, bathymetry=%.5f%n",
            //                 particleId, z, bathymetry);
            //System.out.printf("After scaling → Particle %d: dx=%.5f, dy=%.5f, dz=%.5f%n",
           //                  particleId, dx, dy, dz);


            particle.increment(new double[]{dx, dy, dz});
            
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Error during LLM particle movement execution.");
        }
    }

    @Override
    public void init(IParticle particle) {
        // No specific initialization needed
    }

    @Override
    public void finalize() throws Throwable {
        super.finalize();
    }
}

