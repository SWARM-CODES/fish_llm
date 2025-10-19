import numpy as np

class MechanismBehavior:
    def compute_velocity(self, user_velocity):
        """
        Compute swimming velocity components with randomness.
        """
        k = np.linalg.norm(user_velocity) * np.random.uniform(0, 2)  # Random multiplier for velocity magnitude
        k_prime = np.random.uniform(0, 1)  # Random modifier for direction
        eta = np.random.choice([-1, 1])  # Random sign for U direction
        eta_prime = np.random.choice([-1, 1])  # Random sign for V direction

        U_new = k_prime * eta * k  # Updated U velocity (independent of flow)
        V_new = eta_prime * np.sqrt(abs(k**2 - U_new**2))  # Updated V velocity

        return U_new, V_new, 0  # dz is always 0

    def compute_rheotaxis_orientation(self, flow_velocity):
        """
        Adjust movement direction based on rheotaxis orientation.
        """
        if np.linalg.norm(flow_velocity) == 0:
            return None
        theta_current = np.arctan2(flow_velocity[1], flow_velocity[0])  # Flow direction angle
        theta_direction = theta_current + np.pi  # Opposing direction (rheotaxis)
        theta_rheo = np.random.vonmises(theta_direction, 2)  # Sample from Von Mises distribution
        return None #return none to close rheotaxis, theta_rheo

    def compute_reef_orientation(self, position, reef_positions, detection_radius):
        """
        Adjust movement direction toward the nearest reef if within detection radius.
        """
        closest_reef = None
        min_distance = float('inf')
        for reef in reef_positions:
            distance = np.linalg.norm(position - reef)
            if distance < min_distance:
                min_distance = distance
                closest_reef = reef

        if closest_reef is not None and min_distance < detection_radius:
            direction_to_reef = (closest_reef - position) / min_distance  # Unit vector toward reef
            theta_reef = np.arctan2(direction_to_reef[1], direction_to_reef[0])
            return None #return none to close reef, theta_reef
        return None  # No reef detected within range

    def mechanism_particle_behavior(self, particle_states, reef_positions, detection_radius, current_day):
        """
        Update particle behavior based on mechanism-based swimming model.
        """
        behaviors = []
        egg_stage = (0 <= current_day <= 2)

        for state in particle_states:
            if egg_stage:
               behaviors.append([0.0, 0.0, 0.0])
               continue

            x, y, z = state["x"], state["y"], state["z"]
            u_vel, v_vel = state["u"], state["v"]

            # Compute swimming velocity (independent of flow)
            U_new, V_new, dz = self.compute_velocity(user_velocity=0.4)
            swimming_velocity = np.array([U_new, V_new])

            # Apply rheotaxis
            flow_velocity = np.array([u_vel, v_vel])
            theta_rheo = self.compute_rheotaxis_orientation(flow_velocity)

            # Apply reef orientation if within range
            position = np.array([x, y])
            theta_reef = self.compute_reef_orientation(position, reef_positions, detection_radius)

            # Determine final movement direction
            if theta_rheo is None and theta_reef is None:
                U_final, V_final = U_new, V_new  # Use default swimming velocity
            else:
                theta_final = theta_rheo if theta_rheo is not None else 0  # Default to 0 if none
                if theta_reef is not None:
                    theta_final = (theta_final + theta_reef) / 2  # Blend if both exist

                speed = np.linalg.norm([U_new, V_new])  # Compute absolute velocity magnitude
                U_final = speed * np.cos(theta_final)
                V_final = speed * np.sin(theta_final)
           # U_final = 0.0
           # V_final = 0.0
            behaviors.append([U_final, V_final, dz])

        return np.array(behaviors)

# Example usage:
#mechanism_model = MechanismBehavior()
#particle_behavior = mechanism_model.mechanism_particle_behavior(particle_states, target_locations, detection_radius)

