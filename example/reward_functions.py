import numpy as np

def reward_function_1(trajectories_x, trajectories_y, trajectories_z, bathymetry, time_window, target_locations, radius=5.0, depth_threshold=5.0):
    """
    Reward function for fish larvae settlement based on proximity to target locations and bathymetry.
    """
    start_step, end_step = time_window
    rewards = np.zeros(trajectories_x.shape[0])

    for i in range(rewards.size):
        for step in range(start_step, end_step + 1):
            x, y, z = trajectories_x[i, step], trajectories_y[i, step], trajectories_z[i, step]

            for target_x, target_y in target_locations:
                if np.sqrt((x - target_x)**2 + (y - target_y)**2) <= radius:
                    grid_x, grid_y = int(target_x / 10), int(target_y / 10)
                    expected_depth = bathymetry[grid_y, grid_x]
                    if abs(z - expected_depth) <= depth_threshold:
                        rewards[i] = 1
                        break
    return rewards


def reward_function_2(trajectories_x, trajectories_y, trajectories_z, trajectories_T, trajectories_h, 
                      temperature_range=(16, 22), depth_threshold=5.0, bathymetry_range=(-40, -20)):
    # Initialize rewards (default to 0)
    num_particles = trajectories_x.shape[0]
    rewards = np.zeros(num_particles, dtype=int)

    # Check settlement conditions
    valid_bathymetry = (bathymetry_range[0] <= trajectories_h) & (trajectories_h <= bathymetry_range[1])
    close_to_depth = np.abs(trajectories_z - trajectories_h) <= depth_threshold
    valid_temperature = (temperature_range[0] <= trajectories_T) & (trajectories_T <= temperature_range[1])

    # Assign reward if all conditions are met
    rewards[valid_bathymetry & close_to_depth & valid_temperature] = 1

    return rewards

