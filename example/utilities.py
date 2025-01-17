import numpy as np


from netCDF4 import Dataset

def load_environment(day):
    """
    Load precomputed environmental factors for a specific day.
    """
    with Dataset("environment_data.nc", "r") as nc:
        u_velocity = nc.variables["u_velocity"][day, :, :, :]
        v_velocity = nc.variables["v_velocity"][day, :, :, :]
        w_velocity = nc.variables["w_velocity"][day, :, :, :]
        temperature = nc.variables["temperature"][day, :, :, :]
        bathymetry = nc.variables["bathymetry"][:, :]
    return u_velocity, v_velocity, w_velocity, temperature, bathymetry


def apply_boundary_conditions(x_pos, y_pos, z_pos, depth_pos, domain_length_x, domain_length_y):
    if x_pos < 0:
        x_pos = 0
    elif x_pos > domain_length_x:
        x_pos = domain_length_x

    if y_pos < 0:
        y_pos = 0
    elif y_pos > domain_length_y:
        y_pos = domain_length_y

    if z_pos > 0:
        z_pos = 0  # Stick to surface
    elif z_pos < depth_pos:
        z_pos = depth_pos  # Stick to bottom

    return x_pos, y_pos, z_pos

def get_particle_velocity(x, y, z, daily_u_velocity, daily_v_velocity, daily_w_velocity, domain_depth, x_dim, y_dim, z_dim):
    x_idx = min(max(int(x / 10), 0), x_dim - 1)  # Clamp x index
    y_idx = min(max(int(y / 10), 0), y_dim - 1)  # Clamp y index
    z_idx = min(max(int(z / -domain_depth * z_dim), 0), z_dim - 1)  # Clamp z index

    # Retrieve velocities at the clamped indices
    u_vel = daily_u_velocity[x_idx, y_idx, z_idx]
    v_vel = daily_v_velocity[x_idx, y_idx, z_idx]
    w_vel = daily_w_velocity[x_idx, y_idx, z_idx]

    return u_vel, v_vel, w_vel

def get_bathymetry(x, y, bathymetry, domain_length_x, domain_length_y, x_dim, y_dim):
    x_idx = min(max(int(x / (domain_length_x / x_dim)), 0), x_dim - 1)  # Clamp x index
    y_idx = min(max(int(y / (domain_length_y / y_dim)), 0), y_dim - 1)  # Clamp y index

    
    return bathymetry[y_idx, x_idx]

def get_particle_states(
    x_positions, y_positions, z_positions,
    u_velocity, v_velocity, w_velocity,
    temperature_field_with_noise, bathymetry, sigma_layers,
    domain_length_x, domain_length_y, domain_depth, x_dim, y_dim, z_dim
):
    num_particles = len(x_positions)
    particle_states = np.zeros((num_particles, 7))  # [x, y, z, u, v, w, temp]

    for i in range(num_particles):
        x, y, z = x_positions[i], y_positions[i], z_positions[i]

        # Get velocity at particle location
        u_vel, v_vel, w_vel = get_particle_velocity(
            x, y, z, u_velocity, v_velocity, w_velocity,
            domain_depth, x_dim, y_dim, z_dim
        )

        # Get bathymetry depth and temperature
        x_idx = min(int(x / (domain_length_x / x_dim)), x_dim - 1)
        y_idx = min(int(y / (domain_length_y / y_dim)), y_dim - 1)
        bathymetry_depth = max(abs(bathymetry[y_idx, x_idx]), 1e-5)
        z_idx = int((z / bathymetry_depth) * z_dim)
        z_idx = max(0, min(z_idx, z_dim - 1))

        temperature = temperature_field_with_noise[x_idx, y_idx, z_idx]

        # Store particle state
        particle_states[i] = [x, y, z, u_vel, v_vel, w_vel, temperature]

    return particle_states
def hydrodynamic_and_behavior_update(
    num_particles, steps_per_day, dt, daily_u_velocity, daily_v_velocity, daily_w_velocity,
    x_positions, y_positions, z_positions, particle_behavior, bathymetry,
    trajectories_x, trajectories_y, trajectories_z, trajectories_bathy,
    domain_length_x, domain_length_y, domain_depth, x_dim, y_dim, z_dim, step_index
):
    #step_index = 0

    for step in range(steps_per_day):
        for i in range(num_particles):
            # Current particle positions
            x_pos, y_pos, z_pos = x_positions[i], y_positions[i], z_positions[i]

            # Hydrodynamic velocity
            u_vel, v_vel, w_vel = get_particle_velocity(
                x_pos, y_pos, z_pos, daily_u_velocity, daily_v_velocity, daily_w_velocity,
                domain_depth, x_dim, y_dim, z_dim
            )

            dx_env = u_vel * dt / 1000
            dy_env = v_vel * dt / 1000
            dz_env = w_vel * dt

            # Behavior velocity
            dx_behavior = particle_behavior[i, 0] * dt / 1000
            dy_behavior = particle_behavior[i, 1] * dt / 1000
            dz_behavior = particle_behavior[i, 2] * dt

            # Update positions
            x_pos += dx_env + dx_behavior
            y_pos += dy_env + dy_behavior
            z_pos += dz_env + dz_behavior

            # Store trajectory and apply boundary conditions
            trajectories_bathy[i, step_index] = get_bathymetry(x_pos, y_pos, bathymetry, domain_length_x, domain_length_y, x_dim, y_dim)
            x_positions[i], y_positions[i], z_positions[i] = apply_boundary_conditions(
                x_pos, y_pos, z_pos, trajectories_bathy[i, step_index], domain_length_x, domain_length_y
            )

            trajectories_x[i, step_index] = x_positions[i]
            trajectories_y[i, step_index] = y_positions[i]
            trajectories_z[i, step_index] = z_positions[i]

        step_index += 1

    return trajectories_x, trajectories_y, trajectories_z, trajectories_bathy, step_index

