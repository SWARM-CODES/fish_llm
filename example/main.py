import time
import json
import numpy as np
from setup_ptraj import initialize_particles, initialize_trajectories
from reward_functions import reward_function_1, reward_function_2
from utilities import (
    apply_boundary_conditions,
    get_particle_states,
    get_enabled_states,
    hydrodynamic_and_behavior_update,
    load_environment,
)
from llm_module import LLMBehaviorAPI
from netCDF4 import Dataset
from batchmake import estimate_batch_size, divide_particles_into_batches

# Initialize environment
x_dim = 50   # 50 grid points in x-direction
y_dim = 50   # 50 grid points in y-direction
z_dim = 20   # 20 sigma layers
grid_resolution = 10
sigma_layers = np.linspace(0, -1, z_dim)

# Particle tracking setup
num_particles, num_days, dt = 10, 10, 14400 #dt in seconds
steps_per_day = int(86400 / dt)
total_steps = num_days * steps_per_day

#estimating batch_size
batch_size=5
max_particles_per_batch = estimate_batch_size("prompt.txt", batch_size=batch_size)
#particle_batches = divide_particles_into_batches(num_particles, max_particles_per_batch)


domain_length_x, domain_length_y, domain_depth = 500, 500, 100

x_positions, y_positions, z_positions = initialize_particles(num_particles, domain_length_x, domain_length_y)
trajectories_x, trajectories_y, trajectories_z, trajectories_bathy = initialize_trajectories(num_particles, total_steps)



# Initialize LLM API
llm_api = LLMBehaviorAPI(config_path="config.json")

# Run simulation
target_locations = [(50, 441), (60, 411)]
time_window = (48, 59) #(steps_per_day * 25, steps_per_day * 30)

particle_state = []
history_states = []
trajectory_rewards = []
num_iterations = 1



trajectories_file = "trajectories.nc"
with Dataset(trajectories_file, "w", format="NETCDF4") as nc_file:
    # Define dimensions
    nc_file.createDimension("iteration", num_iterations)
    nc_file.createDimension("times", total_steps)
    nc_file.createDimension("particles", num_particles)

    # Define variables
    x_var = nc_file.createVariable("x", "f4", ("iteration", "particles", "times"))
    y_var = nc_file.createVariable("y", "f4", ("iteration", "particles", "times"))
    z_var = nc_file.createVariable("z", "f4", ("iteration", "particles", "times"))
    bathy_var = nc_file.createVariable("bathy", "f4", ("iteration", "particles", "times"))

    # Add attributes (optional)
    x_var.units = "km"
    y_var.units = "km"
    z_var.units = "m"
    bathy_var.units = "m"
    nc_file.description = "Trajectories of particles for all iterations"
    nc_file.history = "Created using Python NetCDF4 library"
    nc_file.source = "Particle tracking simulation"




# Main loop
for ite in range(1, num_iterations + 1):
    print(f"Iteration {ite}/{num_iterations}")
    start_time = time.time()

    step_index = 0
    x_positions = x_positions.copy()
    y_positions = y_positions.copy()
    z_positions = z_positions.copy()

    # Reset trajectories for each iteration
    trajectories_x[:, :] = np.nan
    trajectories_y[:, :] = np.nan
    trajectories_z[:, :] = np.nan
    trajectories_bathy[:, :] = np.nan

    x_positions, y_positions, z_positions = initialize_particles(num_particles, domain_length_x, domain_length_y)
    trajectories_x[:, 0] = x_positions
    trajectories_y[:, 0] = y_positions
    trajectories_z[:, 0] = z_positions
    
    counter = 0
    iteration_explanations = []
    for day in range(num_days):
        print(f"Day {day + 1}/{num_days}")
        # Generate daily velocity fields
        #if day % 6 == 0:
        counter+=1
        daily_u_velocity, daily_v_velocity, daily_w_velocity, temperature_field_with_noise, bathymetry = load_environment(counter-1)


        # Get particle states
        particle_states = get_particle_states(
            x_positions,
            y_positions,
            z_positions,
            daily_u_velocity,
            daily_v_velocity,
            daily_w_velocity,
            temperature_field_with_noise,
            bathymetry,
            sigma_layers,
            domain_length_x,
            domain_length_y,
            domain_depth,
            x_dim,
            y_dim,
            z_dim,
            day,
        )

        # Initialize history states and rewards if first iteration
        if len(history_states) < num_particles:
            history_states.extend([[] for _ in range(num_particles)])
            trajectory_rewards.extend([[] for _ in range(num_particles)])
        enabled_states = get_enabled_states()
        # Update history states with particle information
        for i in range(num_particles):
            particle_state = particle_states[i]

        # Dynamically create history entry for all enabled states
            history_entry = {"ite": ite}  # Add iteration number
            for key in enabled_states.keys():
                history_entry[key] = particle_state[key]

            history_states[i].append(history_entry)

        # Update particle behavior using LLM
        particle_behavior = llm_api.update_particle_behavior(
                particle_states, history_states, batch_size
        )
        explanations = llm_api.summarize_movements(particle_states, history_states, particle_behavior)
        iteration_explanations.append(explanations)

        # Update particle trajectories with hydrodynamics and LLM behavior
        (
            trajectories_x,
            trajectories_y,
            trajectories_z,
            trajectories_bathy,
            step_index,
        ) = hydrodynamic_and_behavior_update(
            num_particles,
            steps_per_day,
            dt,
            daily_u_velocity,
            daily_v_velocity,
            daily_w_velocity,
            x_positions,
            y_positions,
            z_positions,
            particle_behavior,
            bathymetry,
            trajectories_x,
            trajectories_y,
            trajectories_z,
            trajectories_bathy,
            domain_length_x,
            domain_length_y,
            domain_depth,
            x_dim,
            y_dim,
            z_dim,
            step_index,
        )
        
        # Calculate rewards based on the simulation
    #rewards = reward_function_1(
    #    trajectories_x,
    #    trajectories_y,
    #    trajectories_z,
    #    bathymetry,
    #    time_window,
    #    target_locations,
    #)
    rewards = reward_function_2(
        trajectories_x,
        trajectories_y,
        trajectories_z,
        temperature_field_with_noise,
        bathymetry,
        sigma_layers,
        time_window,
        domain_length_x, 
        domain_length_y, 
        x_dim, 
        y_dim,
        z_dim,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Iteration {ite} completed in {elapsed_time:.2f} seconds.")
    # Update history states with rewards
    for i in range(num_particles):
        trajectory_rewards[i].append(rewards[i])
        history_states[i][-1]["reward"] = rewards[i]

    print(f"Iteration {ite} complete. Rewards: {rewards}")

    # Save iteration-specific explanations
    explanations_file = f"iteration_{ite}_explanations.json"
    with open(explanations_file, "w") as file:
        json.dump(iteration_explanations, file, indent=4)
    print(f"Explanations for iteration {ite} saved to '{explanations_file}'.")

    # Save iteration-specific trajectories
     # Assign data to variable
    with Dataset(trajectories_file, "a") as nc_file:
        x_var = nc_file.variables["x"]
        y_var = nc_file.variables["y"]
        z_var = nc_file.variables["z"]
        bathy_var = nc_file.variables["bathy"]
        x_var[ite-1, :, :] = trajectories_x
        y_var[ite-1, :, :] = trajectories_y
        z_var[ite-1, :, :] = trajectories_z
        bathy_var[ite-1, :, :] = trajectories_bathy



# End of simulation
# Save explanations to a JSON file
#with open("explanations_log.json", "w") as file:
#    json.dump(explanations_log, file, indent=4)
#print("Explanations saved to 'explanations_log.json'.")

# Save final trajectories
#final_trajectories = {
#    "x": trajectories_x.tolist(),
#    "y": trajectories_y.tolist(),
#    "z": trajectories_z.tolist(),
#    "bathy": trajectories_bathy.tolist(),
#}
#with open("final_trajectories.json", "w") as file:
#    json.dump(final_trajectories, file, indent=4)
#print("Final trajectories saved to 'final_trajectories.json'.")

print("Simulation completed.")


