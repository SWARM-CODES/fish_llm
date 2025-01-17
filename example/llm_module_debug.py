import openai
import json
import numpy as np

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

    def update_particle_behavior(self, particle_states, history_states, prompt_path="prompt.txt"):
        # Read the prompt from an external text file
        with open(prompt_path, "r") as f:
            base_prompt = f.read()

        num_particles = particle_states.shape[0]
        behaviors = []
        explanations = []

        for i in range(num_particles):
             behaviors.append([0.0, 0.0, 0.0]) #this is for debug process
             explanations.append("Example explanation for particle movement.") #this is for debug process
      #      x, y, z, u, v, w, temp = particle_states[i]
      #      particle_history = history_states[i] if len(history_states) > i else []

            # Generate particle history string
      #      history_str = "\n".join([
      #          f"Iteration {entry['ite']}, Step {idx + 1}: "
      #          f"X={entry['position'][0]:.2f} km, Y={entry['position'][1]:.2f} km, Z={entry['position'][2]:.2f} m, "
      #          f"Temp={entry['temperature']:.2f} °C, Reward={entry.get('reward', 'N/A')}"
      #          for idx, entry in enumerate(particle_history)
      #      ])

            # Format the prompt
      #      prompt = base_prompt.format(
      #          x=x, y=y, z=z, u=u, v=v, w=w, temp=temp, history_str=history_str
      #      )

            # Make API call with retries
      #      retries = 0
      #      max_retries = 3
      #      while retries < max_retries:
      #          try:
      #              response = self.client.chat.completions.create(
      #                  model=self.model,
      #                  messages=[{"role": "user", "content": prompt}],
      #                  max_tokens=150,
      #                  temperature=0.7
      #              )
      #              response_text = response.choices[0].message.content.strip()

                    # Parse response
      #              movement_line = [line for line in response_text.split("\n") if line.startswith("Movement Vector:")][0]
      #              explanation_line = [line for line in response_text.split("\n") if line.startswith("Explanation:")][0]

      #              dx, dy, dz = map(float, movement_line.replace("Movement Vector:", "").split(","))
      #              explanation = explanation_line.replace("Explanation:", "").strip()

      #              behaviors.append([dx, dy, dz])
      #              explanations.append(explanation)
      #              break
      #          except Exception as e:
      #              retries += 1
      #              if retries == max_retries:
      #                  dx, dy, dz = 0, 0, 0
      #                  explanation = "API call failed."
      #                  behaviors.append([dx, dy, dz])
      #                  explanations.append(explanation)

        return np.array(behaviors), explanations

