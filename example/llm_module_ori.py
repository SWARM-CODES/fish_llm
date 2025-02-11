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
    def call_llm_batch(self, client, model, batch_prompts):
        """
        Call the LLM API with a batch of particle prompts.
        """
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You must respond exactly in the specified format. Deviating from the format is not allowed. Each message relates to one particle."}
            ] + batch_prompts,
            max_tokens=150 * len(batch_prompts),  # Adjust based on batch size
            temperature=0.7
        )
        return response


    def generate_summary_prompt(self, particle_states, history_states, movements, summary_prompt_path):
        """
        Create a prompt for summarizing particle movements.
        """
        # Load the base summary prompt template
        with open(summary_prompt_path, "r") as f:
            base_prompt = f.read()

    
            # Generate dynamic particle states for all particles
        particle_summaries = []
        for i in range(len(particle_states)):
            # Generate the dynamic state for the particle
            dynamic_particle_state = self.generate_dynamic_particle_state(particle_states[i])

            # Extract movement vector
            movement = movements[i]
            dx, dy, dz = movement[0], movement[1], movement[2]

            # Generate history string
            if history_states[i]:
                history_str = "\n".join(
                    [
                        f"    Iteration {entry['ite']}: "
                        + ", ".join([f"{key.capitalize()}={entry[key]:.2f}" for key in entry if key != "ite"])
                        for entry in history_states[i]
                    ]
                )
            else:
                history_str = "    None"

            # Combine the particle data into a single string
            particle_summary = (
                f"Particle {i + 1}:\n"
                f"{dynamic_particle_state}\n"
                f"- Movement Vector:\n"
                f"    - dx: {dx:.2f}, dy: {dy:.2f}, dz: {dz:.2f}\n"
                f"- History:\n"
                f"{history_str}"
            )
            particle_summaries.append(particle_summary)

        # Join all particle summaries with spacing
        dynamic_particle_states = "\n\n".join(particle_summaries)


        # Format the summary prompt
        summary_prompt = base_prompt.replace("{dynamic_particle_states}", dynamic_particle_states)
        return summary_prompt





    def update_particle_behavior(self, particle_states, history_states, batch_size, prompt_path="prompt.txt"):
        # Read the prompt from an external text file
        from ptrajstates_config import PARTICLE_STATE_CONFIG
        enabled_states = get_enabled_states()
        num_particles = len(particle_states)
        behaviors = []
        explanations = []

        for i in range(0, num_particles, batch_size):
      #       behaviors.append([0.0, 0.0, 0.0]) #this is for debug process
      #       explanations.append("Example explanation for particle movement.") #this is for debug process
           batch_particles = particle_states[i:i + batch_size]
           batch_histories = history_states[i:i + batch_size]
            

           # Prepare batch prompts
           batch_prompts = []
           for j, particle_state in enumerate(batch_particles):
               particle_history = batch_histories[j] if j < len(batch_histories) else []

            # Generate particle history string
               history_str = "\n".join([
                   f"Iteration {entry['ite']}, Step {idx + 1}: "
                   + ", ".join([f"{key.capitalize()}={entry[key]:.2f}" for key in entry if key != "ite"])
                   for idx, entry in enumerate(particle_history)
               ])

            # Format the prompt
               prompt = self.generate_prompt(particle_state, history_str, prompt_path)
               batch_prompts.append({"role": "user", "content": prompt})

            # Make API call with retries
           retries = 0
           max_retries = 3
           while retries < max_retries:
               try:
                    response = self.call_llm_batch(self.client, self.model, batch_prompts)
                    for choice in response.choices:
                        response_text = choice.message.content.strip()
                        print("Raw Response:", response_text)
                        try:
                     # Split and parse movement vector
                            lines = response_text.split("\n")
                            if not lines[0].startswith("- Movement Vector:"):
                                raise ValueError("First line must start with 'Movement Vector:'")
                            movement_vector = lines[0].replace("- Movement Vector:", "").strip()
                            dx, dy, dz = map(float, movement_vector.split(","))
                            behaviors.append([dx, dy, dz])
                        except Exception as e:
                            print(f"Error parsing response for a particle: {e}")
                            behaviors.append([0.0, 0.0, 0.0])  # Default fallback
                  #      except Exception as e:
                #response = [{"dx": 0.1, "dy": 0.2, "dz": -0.1} for _ in batch_prompts]
                
                    break
               except Exception:
                retries += 1
                time.sleep(2 ** retries)  # Exponential backoff
                error_message = str(e)
                print(f"Error in batch API call (attempt {retries}/{max_retries}): {error_message}")
                if retries == max_retries:
                     print("API call failed after maximum retries. Using default behavior for the batch.")
                     behaviors.extend([[0.0, 0.0, 0.0]] * len(batch_prompts))
        return np.array(behaviors)
    def summarize_movements(self, particle_states, history_states, movements, summary_prompt_path="summary_prompt.txt"):
        """
        Call an agent to summarize particle movements based on their states, histories, and movements.
        """
        # Generate a summary prompt
        summary_prompt = self.generate_summary_prompt(particle_states, history_states, movements, summary_prompt_path)

        # Make the API call for summary
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You must respond with a summary explanation based on provided data."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=300,  # Adjust tokens for summary response
                temperature=0.7
            )
            summary = response.choices[0].message.content.strip()
           #  print("Raw Response 1:", response)
           #  summary = "Example explanation for particle movement." #this is for debug process
        except Exception as e:
            summary = f"Error generating summary: {e}"

        return summary


