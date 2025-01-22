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


def reward_function_2(trajectories_x, trajectories_y, trajectories_z, temperature_field_with_noise, bathymetry, sigma_layers, time_window, domain_length_x, domain_length_y, x_dim, y_dim, z_dim, temperature_range=(16, 20), depth_threshold=5.0, bathymetry_range=(-40,-20)):
    """
    Reward function for fish larvae settlement based on environmental conditions like temperature and bathymetry.
    """
    start_step, end_step = time_window
    rewards = np.zeros(trajectories_x.shape[0])

    for i in range(rewards.size):
        settled = False
        for step in range(start_step, end_step + 1):
            x, y, z = trajectories_x[i, step], trajectories_y[i, step], trajectories_z[i, step]

            x_idx = min(int(x / (domain_length_x / x_dim)), x_dim - 1)
            y_idx = min(int(y / (domain_length_y / y_dim)), y_dim - 1)
            expected_depth = max(abs(bathymetry[x_idx, y_idx]), 1e-5)
            expected_depth = -expected_depth
            z_idx = int((z / expected_depth) * z_dim)
            z_idx = max(0, min(z_idx, z_dim - 1))
            temperature = temperature_field_with_noise[x_idx, y_idx, z_idx]

            if bathymetry_range[0] <= expected_depth <= bathymetry_range[1]:
                # Check if the particle is near the expected depth
                if abs(z - expected_depth) <= depth_threshold: 
                    # Check if the temperature is within the suitable range
                    if temperature_range[0] <= temperature <= temperature_range[1]:
                        settled = True
                        break  # Stop checking once settled


        rewards[i] = 1 if settled else 0
    return rewards

