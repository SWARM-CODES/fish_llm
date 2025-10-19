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
        coral_signal = nc.variables["coral_signal"][:, :]
        # Transform from (y,x) to (x,y)
        u_velocity = np.transpose(u_velocity, (1, 0, 2))
        v_velocity = np.transpose(v_velocity, (1, 0, 2))
        w_velocity = np.transpose(w_velocity, (1, 0, 2))
        temperature = np.transpose(temperature, (1, 0, 2))
        bathymetry = np.transpose(bathymetry, (1, 0))
        coral_signal = np.transpose(coral_signal, (1, 0))
    return u_velocity, v_velocity, w_velocity, temperature, bathymetry, coral_signal


def apply_boundary_conditions(x_pos, y_pos, z_pos, depth_pos, domain_length_x, domain_length_y):
    if x_pos < 0:
        x_pos = 0
    elif x_pos > domain_length_x:
        x_pos = domain_length_x

    if y_pos < 0:
        y_pos = 0
    elif y_pos > domain_length_y:
        y_pos = domain_length_y
    #print(f"z_pos={z_pos}, depth_pos={depth_pos}")
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

def get_particle_temp(x, y, z, daily_temp, domain_depth, x_dim, y_dim, z_dim):
    x_idx = min(max(int(x / 10), 0), x_dim - 1)  # Clamp x index
    y_idx = min(max(int(y / 10), 0), y_dim - 1)  # Clamp y index
    z_idx = min(max(int(z / -domain_depth * z_dim), 0), z_dim - 1)  # Clamp z index
    
    temp_vel = daily_temp[x_idx, y_idx, z_idx]

    return temp_vel

def get_bathymetry(x, y, bathymetry, domain_length_x, domain_length_y, x_dim, y_dim):
    x_idx = min(max(int(x / (domain_length_x / x_dim)), 0), x_dim - 1)  # Clamp x index
    y_idx = min(max(int(y / (domain_length_y / y_dim)), 0), y_dim - 1)  # Clamp y index

    
    return bathymetry[x_idx, y_idx]

def get_particle_states(
    x_positions, y_positions, z_positions,
    u_velocity, v_velocity, w_velocity,
    temperature_field_with_noise, bathymetry, coral_signal, sigma_layers,
    domain_length_x, domain_length_y, domain_depth, x_dim, y_dim, z_dim, day
):
    enabled_states = get_enabled_states()
    num_particles = len(x_positions)
    particle_states = [enabled_states.copy() for _ in range(num_particles)]

    for i in range(num_particles):
        state = {}
        x, y, z = x_positions[i], y_positions[i], z_positions[i]
        
        if "x" in enabled_states:
                state["x"] = x
        if "y" in enabled_states:
                state["y"] = y
        if "z" in enabled_states:
                state["z"] = z


        # Get velocity at particle location
        u_vel, v_vel, w_vel = get_particle_velocity(
            x, y, z, u_velocity, v_velocity, w_velocity,
            domain_depth, x_dim, y_dim, z_dim
        )
        if "u" in enabled_states:
                state["u"] = u_vel
        if "v" in enabled_states:
                state["v"] = v_vel
        if "w" in enabled_states:
                state["w"] = w_vel

        # Get bathymetry depth and temperature
        x_idx = min(int(x / (domain_length_x / x_dim)), x_dim - 1)
        y_idx = min(int(y / (domain_length_y / y_dim)), y_dim - 1)
        if "bathymetry" in enabled_states:
            bathymetry_depth = max(abs(bathymetry[x_idx, y_idx]), 1e-5)
            state["bathymetry"] = -bathymetry_depth
        z_idx = int((z / bathymetry_depth) * z_dim)
        z_idx = max(0, min(z_idx, z_dim - 1))
        if "coral_signal" in enabled_states:
            state["coral_signal"] = coral_signal[x_idx, y_idx]

        if "temperature" in enabled_states:
            temperature = temperature_field_with_noise[x_idx, y_idx, z_idx]
            state["temperature"] = temperature
        
        if "day" in enabled_states:
           day = day
           state["day"] = day 
        # Store particle state
        particle_states[i].update(state)

    return particle_states
def hydrodynamic_and_behavior_update(
    num_particles, dt, daily_u_velocity, daily_v_velocity, daily_w_velocity, daily_temp,
    x_positions, y_positions, z_positions, particle_behavior, bathymetry,
    trajectories_x, trajectories_y, trajectories_z, trajectories_bathy, trajectories_temp, 
    domain_length_x, domain_length_y, domain_depth, x_dim, y_dim, z_dim, step_index
):
    #step_index = 0

        for i in range(num_particles):
            # Current particle positions
            x_pos, y_pos, z_pos = x_positions[i], y_positions[i], z_positions[i]

            # Hydrodynamic velocity
            u_vel, v_vel, w_vel = get_particle_velocity(
                x_pos, y_pos, z_pos, daily_u_velocity, daily_v_velocity, daily_w_velocity,
                domain_depth, x_dim, y_dim, z_dim
            )

          
            trajectories_temp[i, step_index] = get_particle_temp(x_pos, y_pos, z_pos, daily_temp, domain_depth, x_dim, y_dim, z_dim)

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

            print(
                f"[DEBUG] Particle {i}: "
                f"dx_env={dx_env:.4f}, dy_env={dy_env:.4f}, dz_env={dz_env:.4f} | "
                f"dx_behavior={dx_behavior:.4f}, dy_behavior={dy_behavior:.4f}, dz_behavior={dz_behavior:.4f}"
                )

            # Store trajectory and apply boundary conditions
            trajectories_bathy[i, step_index] = get_bathymetry(x_pos, y_pos, bathymetry, domain_length_x, domain_length_y, x_dim, y_dim)
            x_positions[i], y_positions[i], z_positions[i] = apply_boundary_conditions(
                x_pos, y_pos, z_pos, trajectories_bathy[i, step_index], domain_length_x, domain_length_y
            )

            trajectories_x[i, step_index] = x_positions[i]
            trajectories_y[i, step_index] = y_positions[i]
            trajectories_z[i, step_index] = z_positions[i]

        step_index += 1

        return trajectories_x, trajectories_y, trajectories_z, trajectories_bathy,trajectories_temp, step_index
def get_enabled_states():
    from ptrajstates_config import PARTICLE_STATE_CONFIG
    return {
        key: config["default"]
        for key, config in PARTICLE_STATE_CONFIG.items()
        if config["enabled"]
    }

