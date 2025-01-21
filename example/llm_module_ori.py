import time
import openai
import json
import numpy as np
from utilities import (
    get_enabled_states,
)

class LLMBehaviorAPI:
    def __init__(self, config_path="config.json"):
        # Load configuration
        with open(config_path, "r") as f:
            config = json.load(f)
        self.api_key = config["api_key"]
        self.azure_endpoint = config["azure_endpoint"]
        self.api_version = config["api_version"]
        self.model = config["model"]

        # Initialize OpenAI client
        self.client = openai.AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version
        )
    def generate_dynamic_particle_state(self, particle_state):
        state_lines = []
        for key, value in particle_state.items():
            # Customize formatting for specific states
            if key == "x":
                state_lines.append(f"- Position:\n    - X (East-West, km): {value:.2f} km")
            elif key == "y":
                state_lines.append(f"    - Y (North-South, km): {value:.2f} km")
            elif key == "z":
                state_lines.append(f"    - Z (Depth, m): {value:.2f} m (negative values indicate depth below sea level, so must be negative).")
            elif key == "u":
                state_lines.append(f"- Flow Velocity:\n    - U (East-West speed, m/s): {value:.2f} m/s")
            elif key == "v":
                state_lines.append(f"    - V (North-South speed, m/s): {value:.2f} m/s")
            elif key == "w":
                state_lines.append(f"    - W (Vertical speed, m/s): {value:.2f} m/s")
            elif key == "temperature":
                state_lines.append(f"- Current Temperature: {value:.2f} °C")
            elif key == "bathymetry":
                state_lines.append(f"- Bathymetry Depth: {value:.2f} m")
            elif key == "day":
                state_lines.append(f"- day: {value:.2f} m")
            else:
                state_lines.append(f"- {key.capitalize()}: {value:.2f}")

        return "\n".join(state_lines)
    def generate_prompt(self, particle_state, history_str, base_prompt_path):
    
        # Read the base prompt template
        with open(base_prompt_path, "r") as file:
            base_prompt = file.read()

        # Generate dynamic particle state
        dynamic_particle_state = self.generate_dynamic_particle_state(particle_state)

        # Replace placeholders in the template
        prompt = base_prompt.replace("{dynamic_particle_state}", dynamic_particle_state)
        prompt = prompt.replace("{history_str}", history_str)

        return prompt


    def update_particle_behavior(self, particle_states, history_states, prompt_path="prompt.txt"):
        # Read the prompt from an external text file
        from ptrajstates_config import PARTICLE_STATE_CONFIG
        enabled_states = get_enabled_states()
        num_particles = len(particle_states)
        behaviors = []
        explanations = []

        for i in range(num_particles):
      #       behaviors.append([0.0, 0.0, 0.0]) #this is for debug process
      #       explanations.append("Example explanation for particle movement.") #this is for debug process
            particle_state = particle_states[i]
            particle_history = history_states[i] if len(history_states) > i else []

            # Generate particle history string
            history_str = "\n".join([
            f"Iteration {entry['ite']}, Step {idx + 1}: "
            + ", ".join([f"{key.capitalize()}={entry[key]:.2f}" for key in entry if key != "ite"])
            for idx, entry in enumerate(particle_history)
            ])

            # Format the prompt
            prompt = self.generate_prompt(particle_state, history_str, prompt_path)
           # prompt = base_prompt.format(
           #      history_str=history_str,
           # **{key: particle_state[key] for key in enabled_states.keys()}
           # )

            # Make API call with retries
            retries = 0
            max_retries = 3
            while retries < max_retries:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                        {"role": "system", "content": "You must respond exactly in the specified format. Deviating from the format is not allowed."},
                        {"role": "user", "content": prompt}
                        ],
                        max_tokens=150,
                        temperature=0.7
                    )

                    response_text = response.choices[0].message.content.strip()
                    # print(f"Raw Response from ChatGPT: {response_text}") 
                    # Parse response
                    # Split and validate response lines
                    lines = response_text.split("\n")
                    if len(lines) < 2:
                        raise ValueError("Response does not contain the expected two lines.")

                    # Validate "Movement Vector" line
                    if not lines[0].startswith("- Movement Vector:"):
                        raise ValueError("First line must start with 'Movement Vector:'")
                    # Parse "Movement Vector"
                    movement_vector = lines[0].replace("- Movement Vector:", "").strip()
                    try:
                        dx, dy, dz = map(float, movement_vector.split(","))
                    except ValueError as e:
                        raise ValueError(f"Error parsing Movement Vector: {movement_vector}") from e

        # Validate "Explanation" line
                    if not lines[1].startswith("- Explanation:"):
                        raise ValueError("Second line must start with 'Explanation:'")
                    if "<dx>" in lines[1] or "<dy>" in lines[1] or "<dz>" in lines[1] or "<temp>" in lines[1]:
                        raise ValueError("Placeholders not replaced in Explanation line.")
                    
                    explanation = lines[1].strip().replace("- Explanation:", "").strip()
                    behaviors.append([dx, dy, dz])
                    explanations.append(explanation)
                    break
                except Exception as e:
                    retries += 1
                    wait_time = 2 ** retries  # Exponential backoff
                    print(f"Retry {retries}/{max_retries}. Waiting {wait_time}s due to error: {e}")
                    time.sleep(wait_time)
                    if retries == max_retries:
                        dx, dy, dz = 0, 0, 0
                        explanation = "API call failed."
                        behaviors.append([dx, dy, dz])
                        explanations.append(explanation)

        return np.array(behaviors), explanations

