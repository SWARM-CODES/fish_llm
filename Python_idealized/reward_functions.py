import numpy as np

def reward_function_1(trajectories_x, trajectories_y, trajectories_z, trajectories_h, time_window,
                      target_locations, radius=0.5, depth_threshold=200.0):
    """
    Reward function for fish larvae settlement based on proximity to coral reefs and bathymetry.
    """
    start_step, end_step = time_window
    
    # Initialize rewards (default to 0)
    num_particles = trajectories_x.shape[0]
    rewards = np.zeros(num_particles, dtype=int)
    
    # Compute proximity condition
    for target_x, target_y in target_locations:
        within_radius = np.sqrt((trajectories_x - target_x) ** 2 +
                                (trajectories_y - target_y) ** 2) <= radius
        
        # Compute depth condition
        within_depth = np.abs(trajectories_z - trajectories_h) <= depth_threshold
        
        # Assign reward if both conditions are met
        rewards[within_radius & within_depth] = 1
    
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

