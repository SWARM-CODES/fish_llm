import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import openai



# Define domain parameters
x_dim = 50   # 50 grid points in x-direction
y_dim = 50   # 50 grid points in y-direction
z_dim = 20   # 20 sigma layers
grid_resolution = 10  # 10 km resolution per grid cell
bathymetry_shallow = -10   # Depth at the western edge in meters
bathymetry_deep = -100     # Depth at the eastern edge in meters
surface_velocity = -0.5   # m/s, east-to-west flow at the surface
velocity_decrease = 0.02  # m/s, decrease in velocity per sigma layer to bottom


# Calculate linear increase rate for bathymetry (in meters per grid cell)
increase_rate = 2.0


# Generate grid coordinates
x = np.arange(0, x_dim * grid_resolution, grid_resolution)  # x-coordinates
y = np.arange(0, y_dim * grid_resolution, grid_resolution)  # y-coordinates
sigma_layers = np.linspace(0, -1, z_dim)  # Sigma layers from surface (0) to bottom (-1)


# Create bathymetry (depth) matrix
depth = bathymetry_shallow - np.arange(x_dim) * increase_rate  # Linearly increasing depth from west to east
print(f"The depth {depth} meters.")


bathymetry = np.tile(depth, (y_dim, 1))  # Extend depth profile uniformly in y-direction


# 3D bathymetry grid across sigma layers
z = np.array([bathymetry * sigma for sigma in sigma_layers])


# Create x, y, z matrices representing grid coordinates in 3D
x_grid, y_grid, z_grid = np.meshgrid(x, y, sigma_layers, indexing='ij')




# Create temperature field
surface_temp_min = 20  # Surface temperature at the west
surface_temp_max = 30  # Surface temperature at the east
bottom_temp_min = 10    # Bottom temperature at the east
bottom_temp_max = 20   # Bottom temperature at the west
temperature_field = np.zeros((x_dim, y_dim, z_dim))
surface_temps = np.linspace(surface_temp_min, surface_temp_max, x_dim)  
bottom_temps = np.linspace(bottom_temp_max, bottom_temp_min, x_dim)
# Fill temperature field with linear interpolation between surface and bottom temperatures
for i in range(x_dim):
    for j in range(y_dim):
        temperature_field[i, j, :] = np.linspace(surface_temps[i], bottom_temps[i], z_dim)
# Add random noise to the temperature field
noise = np.random.uniform(-1, 1, (x_dim, y_dim, z_dim))
temperature_field_with_noise = temperature_field + noise






# Initialize velocity fields
u_velocity = np.zeros((x_dim, y_dim, z_dim))  # East-west velocity
v_velocity = np.zeros((x_dim, y_dim, z_dim))  # North-south velocity (set to zero)
w_velocity = np.zeros((x_dim, y_dim, z_dim))  # Vertical velocity (set to zero)


# Set east-to-west flow with linear decrease from surface to bottom
for k in range(z_dim):
    u_velocity[:, :, k] = surface_velocity + k * (velocity_decrease)  # Decrease in speed with depth


# Save information into an xarray Dataset
ds = xr.Dataset(
    {
        "x_grid": (("x", "y", "z"), x_grid),
        "y_grid": (("x", "y", "z"), y_grid),
        "z_grid": (("x", "y", "z"), z_grid),
        "bathymetry": (("x", "y"), bathymetry),
        "u_velocity": (("x", "y", "z"), u_velocity),
        "v_velocity": (("x", "y", "z"), v_velocity),
        "w_velocity": (("x", "y", "z"), w_velocity)
    },
    coords={
        "x": x,
        "y": y,
        "z_sigma": sigma_layers
    }
)


# Particle tracking experiment
num_particles = 2
num_days = 30
dt = 14400  # Time step in seconds (4 hours)
steps_per_day = int(86400 / dt)
total_steps = num_days * steps_per_day
domain_length_x = 500  # Domain length in km
domain_length_y = 500  # Domain length in km
domain_depth = 100 



x_positions = np.full(num_particles, 495) #start at most east grid
y_positions = np.random.uniform(0, domain_length_y, num_particles) #random released in y direction for the mosteast grid  
z_positions = np.zeros(num_particles) #start at surface
particle_behavior = np.zeros((num_particles, 3))  # Behavior speeds for dx, dy, dz




# Initialize arrays to store particle trajectories (in km for x and y, and meters for z)
trajectories_x = np.full((num_particles, total_steps), np.nan)
trajectories_y = np.full((num_particles, total_steps), np.nan)
trajectories_z = np.full((num_particles, total_steps), np.nan)
trajectories_bathy = np.full((num_particles, total_steps), np.nan)


# Store initial positions
trajectories_x[:, 0] = x_positions
trajectories_y[:, 0] = y_positions
trajectories_z[:, 0] = z_positions
trajectories_bathy[:, 0] = -100




# Function to create daily velocity fields with variation
def create_daily_velocity_fields(u_velocity, v_max_variation=0.5):
    # Apply ±10% random variation to u_velocity field
    daily_u_velocity = u_velocity * (1 + np.random.uniform(-0.1, 0.1, u_velocity.shape))
    # Generate a daily v_velocity field with random values between -v_max_variation and +v_max_variation
    daily_v_velocity = np.random.uniform(-v_max_variation, v_max_variation, u_velocity.shape)
    # Vertical velocity remains zero
    daily_w_velocity = np.zeros_like(u_velocity)
    return daily_u_velocity, daily_v_velocity, daily_w_velocity


# Function to get particle velocity based on position
def get_particle_velocity(x_pos, y_pos, z_pos, daily_u_velocity, daily_v_velocity, daily_w_velocity):
    # Convert position to nearest grid indices
    x_idx = min(int(x_pos / (domain_length_x / x_dim)), x_dim - 1)
    y_idx = min(int(y_pos / (domain_length_y / y_dim)), y_dim - 1)
    z_idx = min(int((z_pos / domain_depth) * z_dim), z_dim - 1)
    
    # Return the velocity at the particle's grid location
    u_vel = daily_u_velocity[x_idx, y_idx, z_idx]
    v_vel = daily_v_velocity[x_idx, y_idx, z_idx]
    w_vel = daily_w_velocity[x_idx, y_idx, z_idx]
    return u_vel, v_vel, w_vel


# Function to apply boundary conditions
def apply_boundary_conditions(x_pos, y_pos, z_pos, depth_pos):
    x_pos = max(0, min(x_pos, domain_length_x))
    y_pos = max(0, min(y_pos, domain_length_y))
    z_pos = max(z_pos, depth_pos)
    return x_pos, y_pos, z_pos


# Function to get bathymetry at a particle's position
def get_bathymetry(x_pos, y_pos, bathymetry):
    # Convert position to nearest grid indices
    x_idx = min(int(x_pos / (domain_length_x / x_dim)), x_dim - 1)
    y_idx = min(int(y_pos / (domain_length_y / y_dim)), y_dim - 1)
    
    # Return the bathymetry depth at the particle's grid location
    depth_ptraj = bathymetry[y_idx, x_idx]
    return depth_ptraj


def reward_function_1(
    trajectories_x, trajectories_y, trajectories_z, 
    bathymetry, sigma_layers, x_grid, y_grid, z_grid, 
    target_locations, time_window, radius=5.0, depth_threshold=5.0
):
    """
    Reward function for fish larvae settlement, considering updated bathymetry logic.


    Parameters:
    - trajectories_x, trajectories_y, trajectories_z: 2D arrays of particle trajectories (num_particles, num_steps).
    - bathymetry: 2D array representing the extended depth profile.
    - sigma_layers: 1D array of sigma layer fractions.
    - x_grid, y_grid, z_grid: 3D arrays representing the grid coordinates in 3D space.
    - target_locations: List of tuples, each indicating the (x, y) coordinate of a target location.
    - time_window: Tuple (start_step, end_step) defining the time range to check trajectories.
    - radius: Radius around the target location within which settlement is valid (default: 5.0 meters).
    - depth_threshold: Distance from bathymetry below which particles are considered successfully settled (default: 5.0 meters).


    Returns:
    - rewards: 1D array of rewards (1 for successful settlement, 0 otherwise).
    """
    num_particles, num_steps = trajectories_x.shape
    rewards = np.zeros(num_particles)  # Initialize rewards array


    start_step, end_step = time_window


    # Iterate over particles
    for i in range(num_particles):
        settled = False
        # Check trajectories in the defined time window
        for step in range(start_step, end_step + 1):
            particle_x = trajectories_x[i, step]
            particle_y = trajectories_y[i, step]
            particle_z = trajectories_z[i, step]


            # Check proximity to each target location
            for target_x, target_y in target_locations:
                distance = np.sqrt((particle_x - target_x)**2 + (particle_y - target_y)**2)
                if distance <= radius:
                    # Find the nearest grid point in x, y
                    grid_x_idx = (np.abs(x_grid[:, 0, 0] - target_x)).argmin()
                    grid_y_idx = (np.abs(y_grid[0, :, 0] - target_y)).argmin()


                    # Get the expected depth from bathymetry at the nearest grid point
                    expected_depth = z_grid[grid_x_idx, grid_y_idx, :].min()  # Minimum depth across sigma layers


                    # Check depth criterion
                    if abs(particle_z - expected_depth) <= depth_threshold:
                        settled = True
                        break  # Exit loop if settled


            if settled:
                break  # Exit time loop if settled


        # Assign reward based on settlement status
        rewards[i] = 1 if settled else 0


    return rewards


def reward_function_2(
    trajectories_x, trajectories_y, trajectories_z, 
    temperature_field_with_noise, bathymetry, sigma_layers, 
    time_window, temperature_range=(17, 20), depth_threshold=5.0
):
    """
    Reward function for larval settlement based on environmental conditions.


    Parameters:
    - trajectories_x, trajectories_y, trajectories_z: 2D arrays of particle trajectories (num_particles, num_steps).
    - temperature_field_with_noise: 3D array of temperatures (x_dim, y_dim, z_dim).
    - bathymetry: 2D array representing bathymetry (depth) of the area (x_dim, y_dim).
    - sigma_layers: 1D array of sigma layer fractions used for the 3D bathymetry grid.
    - time_window: Tuple (start_step, end_step) defining the time range to check trajectories.
    - temperature_range: Tuple (min_temp, max_temp) defining the suitable temperature range.
    - depth_threshold: Distance from bathymetry below which larvae are considered settled (default: 5.0 meters).


    Returns:
    - rewards: 1D array of rewards (1 for successful settlement, 0 otherwise).
    """
    num_particles, num_steps = trajectories_x.shape
    rewards = np.zeros(num_particles)  # Initialize rewards array


    start_step, end_step = time_window


    # Iterate over particles
    for i in range(num_particles):
        settled = False
        # Check trajectories in the defined time window
        for step in range(start_step, end_step + 1):
            particle_x = trajectories_x[i, step]
            particle_y = trajectories_y[i, step]
            particle_z = trajectories_z[i, step]


            # Find the nearest grid point in x, y
            grid_x_idx = (np.abs(np.arange(bathymetry.shape[0]) - particle_x)).argmin()
            grid_y_idx = (np.abs(np.arange(bathymetry.shape[1]) - particle_y)).argmin()


            # Calculate expected bathymetry at the particle's location
            expected_depth = bathymetry[grid_x_idx, grid_y_idx]


            # Check depth criterion
            if abs(particle_z - expected_depth) <= depth_threshold:
                # Get the temperature at the particle's location and vertical layer
                z_layer_idx = (np.abs(sigma_layers - particle_z / expected_depth)).argmin()
                temperature = temperature_field_with_noise[grid_x_idx, grid_y_idx, z_layer_idx]


                # Check temperature suitability
                if temperature_range[0] <= temperature <= temperature_range[1]:
                    settled = True
                    break  # Exit loop if settled


        # Assign reward based on settlement status
        rewards[i] = 1 if settled else 0


    return rewards







history_states = []

def update_particle_behavior(particle_states, history_states):
    client = openai.AzureOpenAI(
        api_version="2023-05-15",
        azure_endpoint="https://guhuwang.openai.azure.com/",
        api_key="6b2flMrX3OqSmsD3ZQkQLRimXWvzAdf1ydRPpnCGGyuKNTbTAa4VJQQJ99ALACYeBjFXJ3w3AAABACOGwr5s"
    )

    num_particles = particle_states.shape[0]
    behaviors = []
    explanations = []  ### for explanations

    for i in range(num_particles):
        x, y, z, u, v, w, temp = particle_states[i]

         
        particle_history = history_states[i] if len(history_states) > i else []

        
        history_str = "\n".join([
            f"Iteration {entry['ite']}, Step {idx + 1}: "
            f"X={entry['position'][0]:.2f} km, Y={entry['position'][1]:.2f} km, Z={entry['position'][2]:.2f} m, "
            f"Temp={entry['temperature']:.2f} °C, Reward={entry.get('reward', 'N/A')}"
            for idx, entry in enumerate(particle_history)
        ])

        
        prompt = f"""
        You are an agent controlling fish particles in an ocean environment.
        - **Current Particle State**:
            - Position: 
                - X (East-West, km): {x:.2f} km
                - Y (North-South, km): {y:.2f} km
                - Z (Depth, m): {z:.2f} m (negative values indicate depth below sea level, so must be negative)
            - Flow Velocity: 
                - U (East-West speed, m/s): {u:.2f} m/s
                - V (North-South speed, m/s): {v:.2f} m/s
                - W (Vertical speed, m/s): {w:.2f} m/s
            - Current Temperature: {temp:.2f} °C
        - **Larval Attributes**:
            - Preferred Temperature Range: 17-30°C
            - Vertical Movement Rate (OVM): Commonly 1e-4 m/s
            - Swimming Behavior: commonly 0.10 m/s, at this scale
            
        - **Particle Movement History**:
        {history_str}
        ### Task
        1. **Choose a Vertical Migration Pattern**:
            Dynamically select one of the following based on environmental conditions and larval state:
                - **Surface dwelling**:
                    - Favor the upper 10 meters if temperatures are within the preferred range (17-30°C) and settlement cues are distant.
                - **Ontogenetic Vertical Migration (OVM)**:
                    - Gradually adjust depth, averaging 1e-4 m/s, influenced by flow resistance and temperature gradients.
                    - Favor deeper areas as larvae mature.
                - **Wide vertical spread**:
                    - Distribute across depths to maximize dispersal opportunities if environmental conditions vary significantly or
                    settlement cues are unclear.

            **Decision Factors**:
                - Favor regions with temperatures between 17-30°C.

            2. **Determine Swimming Behavior**:
                - **Active Swimming**:
                    - Adjust speed dynamically, commonly 0.10 m/s, based on:
                    - Flow strength (u, v) for efficient movement.
                    - Proximity to settlement areas in the northwest.
                    - Direct swimming to shallower areas, guided by bathymetry changes.
                - **Passive Drift**:
                    - Pre-flexion larvae rely solely on environmental flow (u, v, w).

            3. **Evaluate Environmental Suitability**:
                - Avoid unsuitable temperatures (<17°C or >30°C).
                - Adjust vertical and horizontal movement to optimize settlement chances in the northwest shallow area.

        Decide the next movement as a vector (dx, dy, dz) in m/s for X and Y, and m/s for Z and explain your reasoning.
        
        ### Output Format 
        - Movement Vector: dx, dy, dz
        - Explanation: A short text explaining the reasoning behind this decision. no more than 20 words
        ### example output (must follow this form, do not say any other words, or repalce Movement Vector and Explanation with other words):
        Movement Vector: 0.1, 0.1, 0.0001
        Explanation: A short text explaining
        """
        
        try:
            retries = 0
            max_retries = 3  
            while retries < max_retries:
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150,  
                        temperature=0.7
                    )
                    
                    response_text = response.choices[0].message.content.strip()

                    # Movement Vector: dx, dy, dz
                    # Explanation: <reasoning>
                    lines = response_text.split("\n")

                    movement_line = [line for line in lines if line.startswith("Movement Vector:")][0]
                    explanation_line = [line for line in lines if line.startswith("Explanation:")][0]
                    
                    dx, dy, dz = map(float, movement_line.replace("Movement Vector:", "").split(","))
                    explanation = explanation_line.replace("Explanation:", "").strip()

                    break  #
                except Exception as e:
                    retries += 1
                    print(f"Error with OpenAI API: {e}")
                    if retries < max_retries:
                        print(f"Retrying... ({retries}/{max_retries})")
                      
                    else:
                        print("Max retries reached. Returning default values.")
                        dx, dy, dz = 0, 0, 0
                        explanation = "No explanation due to API error."
                        break
        except Exception as e:
            print(f"Unexpected error: {e}")
            dx, dy, dz = 0, 0, 0
            explanation = "No explanation due to an unexpected error."

        behaviors.append([dx, dy, dz])
        print(explanation)
        explanations.append(explanation)

    return np.array(behaviors)



def get_particle_states(
    x_positions, y_positions, z_positions,
    u_velocity, v_velocity, w_velocity,
    temperature_field_with_noise, bathymetry, sigma_layers
):
    num_particles = len(x_positions)
    particle_states = np.zeros((num_particles, 7))  

    for i in range(num_particles):
       
        x, y, z = x_positions[i], y_positions[i], z_positions[i]
        
      
        u_vel, v_vel, w_vel = get_particle_velocity(x, y, z, u_velocity, v_velocity, w_velocity)

    
        x_idx = min(int(x / (domain_length_x / x_dim)), x_dim - 1)
        y_idx = min(int(y / (domain_length_y / y_dim)), y_dim - 1)
        
        bathymetry_depth = max(abs(bathymetry[y_idx, x_idx]), 1e-5)  
        z_idx = int((z / bathymetry_depth) * z_dim)
        z_idx = max(0, min(z_idx, z_dim - 1)) 
        

        temperature = temperature_field_with_noise[x_idx, y_idx, z_idx]


        particle_states[i] = [x, y, z, u_vel, v_vel, w_vel, temperature]
    
    return particle_states




target_locations = [(50, 441), (60, 411), (65, 400), (45, 488)]
time_window = (149, 179) #step_index =25-30days
# Calculate rewards function 1



num_iterations = 2


history_states = []
trajectory_rewards = []

for ite in range(1, num_iterations + 1):  
    step_index = 0
    for day in range(1, num_days-1):

        daily_u_velocity, daily_v_velocity, daily_w_velocity = create_daily_velocity_fields(u_velocity)


        particle_states = get_particle_states(
            x_positions, y_positions, z_positions,
            daily_u_velocity, daily_v_velocity, daily_w_velocity,
            temperature_field_with_noise, bathymetry, sigma_layers
        )


        if len(history_states) < num_particles:
            history_states.extend([[] for _ in range(num_particles)])
            trajectory_rewards.extend([[] for _ in range(num_particles)])


        for i in range(num_particles):
            x, y, z, u, v, w, temp = particle_states[i]
            history_states[i].append({
                "ite": ite,
                "position": (x, y, z),
                "temperature": temp
            })


        particle_behavior = update_particle_behavior(particle_states, history_states)
        print(particle_behavior)
 
        for step in range(steps_per_day):
            for i in range(num_particles):

                x_pos, y_pos, z_pos = x_positions[i], y_positions[i], z_positions[i]


                u_vel, v_vel, w_vel = get_particle_velocity(
                    x_pos, y_pos, z_pos, daily_u_velocity, daily_v_velocity, daily_w_velocity
                )
                dx_env = u_vel * dt / 1000  
                dy_env = v_vel * dt / 1000  
                dz_env = w_vel * dt         

                
                dx_behavior = particle_behavior[i, 0] * dt / 1000  
                dy_behavior = particle_behavior[i, 1] * dt / 1000  
                dz_behavior = particle_behavior[i, 2] * dt        

                
                x_pos += dx_env + dx_behavior
                y_pos += dy_env + dy_behavior
                z_pos += dz_env + dz_behavior

                
                trajectories_bathy[i, step_index] = get_bathymetry(x_pos, y_pos, bathymetry)

                
                x_positions[i], y_positions[i], z_positions[i] = apply_boundary_conditions(
                    x_pos, y_pos, z_pos, trajectories_bathy[i, step_index]
                )

                
                trajectories_x[i, step_index] = x_positions[i]
                trajectories_y[i, step_index] = y_positions[i]
                trajectories_z[i, step_index] = z_positions[i]

            
            step_index += 1

    
    rewards = reward_function_1(
        trajectories_x, trajectories_y, trajectories_z,
        bathymetry, sigma_layers, x_grid, y_grid, z_grid,
        target_locations, time_window
    )

    
    for i in range(num_particles):
        trajectory_rewards[i].append(rewards[i])  
        history_states[i][-1]["reward"] = rewards[i]  

    
    print(f"Iteration {ite} complete. Rewards: {rewards}")








rewards1 = reward_function_1(
    trajectories_x, trajectories_y, trajectories_z, 
    bathymetry, sigma_layers, x_grid, y_grid, z_grid, 
    target_locations, time_window
)
print("Rewards1:", rewards1)


#rewards2 = reward_function_2(
#    trajectories_x, trajectories_y, trajectories_z,
#    temperature_field_with_noise, bathymetry, sigma_layers,
#    time_window
#)
#print("Rewards2:", rewards2)





# final--ite trajectory
final_trajectories_x = trajectories_x[:, step_index - steps_per_day * num_days:step_index]
final_trajectories_y = trajectories_y[:, step_index - steps_per_day * num_days:step_index]
final_trajectories_z = trajectories_z[:, step_index - steps_per_day * num_days:step_index]
final_trajectories_bathy = trajectories_bathy[:, step_index - steps_per_day * num_days:step_index]

plotsteps1 = final_trajectories_x.shape[1]
plotsteps2 = np.arange(plotsteps1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Subplot 1: XY Trajectory Plot for the final iteration
for i in range(num_particles):
    ax1.plot(final_trajectories_x[i, :], final_trajectories_y[i, :], lw=0.5)  # Particle trajectories
# Plot the target locations as red diamonds
for target in target_locations:
    ax1.scatter(target[0], target[1], marker='D', color='black', s=10)

ax1.set_title("XY Trajectories of Particles (Final Iteration)")
ax1.set_xlabel("X Position (km)")
ax1.set_ylabel("Y Position (km)")
ax1.set_xlim(0, 500)
ax1.set_ylim(0, 500)
ax1.grid()

# Subplot 2: Depth Change with Time for the final iteration
for i in range(num_particles):
    ax2.plot(plotsteps2 * (dt / 3600), final_trajectories_z[i, :], lw=0.5)  # Depth changes

# Fill bathymetry for each particle in the final iteration
for i in range(num_particles):
    ax2.fill_between(plotsteps2 * (dt / 3600), final_trajectories_bathy[i, :], -100, color="gray", alpha=1.0)

ax2.set_title("Depth Change of Particles Over Time (Final Iteration)")
ax2.set_xlabel("Time (hours)")
ax2.set_ylabel("Depth (m)")
ax2.set_ylim(0, -100)
ax2.invert_yaxis()  # Depth increases downward
ax2.grid()

# Adjust layout and display the figure
plt.tight_layout()
plt.show()

# Plot bottom temperature profile as before
bottom_temp_layer = temperature_field_with_noise[:, :, -2]  # Last layer in the z-dimension

# Create contour plot for the bottom sigma layer
fig, ax = plt.subplots(figsize=(10, 8))
contour = ax.contourf(
    np.arange(bottom_temp_layer.shape[0]),
    np.arange(bottom_temp_layer.shape[1]),
    bottom_temp_layer.T,
    cmap='coolwarm', levels=50
)

# Add colorbar
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('Temperature (°C)', fontsize=12)

# Add isothermal lines
isothermal_22 = ax.contour(
    np.arange(bottom_temp_layer.shape[0]),
    np.arange(bottom_temp_layer.shape[1]),
    bottom_temp_layer.T,
    levels=[17], colors='black', linestyles='-', linewidths=1.5
)
isothermal_25 = ax.contour(
    np.arange(bottom_temp_layer.shape[0]),
    np.arange(bottom_temp_layer.shape[1]),
    bottom_temp_layer.T,
    levels=[20], colors='black', linestyles='-', linewidths=1.5
)

# Add labels and title
ax.set_title("Bottom Sigma Layer Temperature Contour", fontsize=14)
ax.set_xlabel("X Grid Points", fontsize=12)
ax.set_ylabel("Y Grid Points", fontsize=12)

plt.tight_layout()
plt.show()
