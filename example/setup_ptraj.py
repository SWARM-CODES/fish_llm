import numpy as np

def initialize_particles(num_particles, domain_length_x, domain_length_y):
    ini_x_positions = np.full(num_particles, domain_length_x - 5)  # Start at most east grid
    ini_y_positions0 = [150, 450] #np.random.uniform(0, domain_length_y, num_particles)  # Random y positions
    ini_y_positions = np.array([ini_y_positions0[0]] * (num_particles // 2) + [ini_y_positions0[1]] * (num_particles // 2))
    #ini_y_positions = np.random.uniform(0, domain_length_y, num_particles)  # Random y positions
    ini_z_positions = np.zeros(num_particles)  # Start at surface
    return ini_x_positions, ini_y_positions, ini_z_positions

def initialize_trajectories(num_particles, total_steps):
    trajectories_x = np.full((num_particles, total_steps), np.nan)
    trajectories_y = np.full((num_particles, total_steps), np.nan)
    trajectories_z = np.full((num_particles, total_steps), np.nan)
    trajectories_bathy = np.full((num_particles, total_steps), np.nan)
    trajectories_temp = np.full((num_particles, total_steps), np.nan)
    return trajectories_x, trajectories_y, trajectories_z, trajectories_bathy, trajectories_temp

