import time
from openai import OpenAI
import json
import tiktoken
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
        self.endpoint = config["endpoint"]
        self.model = config["model"]

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key, base_url=self.endpoint)
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
                state_lines.append(f"    - W (Vertical speed, m/s): {value:.3e} m/s")
            elif key == "temperature":
                state_lines.append(f"- Current Temperature: {value:.2f} °C")
            elif key == "bathymetry":
                state_lines.append(f"- Bathymetry Depth: {value:.2f} m")
            elif key == "day":
                state_lines.append(f"- day: {value:.2f} m")
            elif key == "coral_signal":
                state_lines.append(f"- coral_signal: {value:.3e} ")
            else:
                state_lines.append(f"- {key.capitalize()}: {value:.2f}")
        return "\n".join(state_lines)

    def call_llm_batch(self, particle_states, history_states,batch_size, system_prompt, max_tokens_per_particle=3000, temperature=0.7):
        
        def _get_encoder(model_name: str = "gpt-4o-mini"):
            try:
                return tiktoken.encoding_for_model(model_name)
            except KeyError:
                return tiktoken.get_encoding("cl100k_base")

        def count_chat_tokens_approx(messages, model_name: str = "gpt-4o-mini"):
            enc = _get_encoder(model_name)
            per_message = []
            total = 0
            # Heuristic overhead similar to GPT-3.5/4 chat format
            PER_MSG_OVERHEAD = 4   # tokens per message (role, separators, etc.)
            REPLY_PRIMER = 2       # assistant reply primer

            for m in messages:
                content = m.get("content", "")
                role = m.get("role", "")
                # Count content + role tokens; add overhead
                tokens = len(enc.encode(content)) + len(enc.encode(role)) + PER_MSG_OVERHEAD
                per_message.append(tokens)
                total += tokens
            total += REPLY_PRIMER
            return total, per_message




        num_particles = len(particle_states)
        behaviors = []
        batch_results = []
        #batch_size is not used
        batch_size = num_particles
        for i in range(0, num_particles, batch_size):
            batch_particles = particle_states[i:i + batch_size]
            batch_histories = history_states[i:i + batch_size]
        # one system + many users
            messages = [{"role": "system", "content": system_prompt}]
            for j, particle_state in enumerate(batch_particles):
                particle_history = batch_histories[j] if j < len(batch_histories) else []

                # Your original history string builder (kept as requested)
                history_str = "\n".join([
                    f"Iteration {entry['ite']}, Step {idx + 1}: "
                    + ", ".join([f"{key.capitalize()}={entry[key]:.3e}" for key in entry if key != "ite"])
                    for idx, entry in enumerate(particle_history)
                ])

                # Per-particle content (label with Particle k)
                content = (
                    f"Particle {i + j + 1}:\n"
                    f"STATE:\n{self.generate_dynamic_particle_state(particle_state)}\n\n"
                    f"HISTORY:\n{history_str}\n"
                    "Include this particle's result in the consolidated JSON array, same order as messages."
                )
                messages.append({"role": "user", "content": content})
            # Debug: print constructed messages instead of calling API
            print("=" * 60)
            #print(f"Batch {i//batch_size + 1} messages being sent:")
            batch_total_tokens, per_msg_tokens = count_chat_tokens_approx(messages, model_name=self.model)
            #for m in messages:
                #print(f"[{m['role'].upper()}] {m['content']}\n")
            
            print(f"≈ Batch token total (prompt only): {batch_total_tokens} tokens")
            print("-" * 60)
            # Fake response with dx, dy, dz for each particle
            fake_choices = [
                {"message": {"content": json.dumps([
                    {"particle": k + i + 1, "dx": 0.1, "dy": 0.2, "dz": -0.1}
                    for k in range(len(batch_particles))
                ])}}
            ]

            fake_resp = type("FakeResp", (), {})()   # simple dummy object
            fake_resp.choices = fake_choices

            batch_results.append((i, len(batch_particles), fake_resp))



            #Call the API
            #retries, max_retries = 0, 3
            #while True:
            #    try:
            #        resp = self.client.chat.completions.create(
            #            model=self.model,
            #            messages=messages,
            #            max_tokens=max_tokens_per_particle * len(batch_particles),
            #            temperature=temperature,
            #        )
                    # return the raw response with index info so you can align later
            #        batch_results.append((i0, len(batch_particles), resp))
            #        break  # success
            #    except Exception as e:
            #        retries += 1
            #        print(f"[LLM batch error] attempt {retries}/{max_retries}: {e}")
            #        if retries >= max_retries:
                        # append a sentinel None so downstream can handle fallback
            #            batch_results.append((i0, len(batch_particles), None))
            #            break
            #        time.sleep(2 ** retries)  # backoff

        return batch_results
    
    def parse_llm_batch_responses(self, batch_results, num_particles):

        behaviors = np.zeros((num_particles, 3), dtype=float)

        for i0, cnt, resp in batch_results:
            if resp is None:
                # already logged; leave zeros
                continue

            try:
                text = resp.choices[0].message.content.strip()
                # Expect a JSON array; if wrapped, extract with regex
                try:
                    parsed = json.loads(text)
                    if not isinstance(parsed, list):
                        raise ValueError("Assistant did not return a JSON array.")
                except Exception:
                    m = re.search(r"(\[.*\])", text, flags=re.S)
                    if not m:
                        raise
                    parsed = json.loads(m.group(1))

            # map ordered results into the correct slice
                for k in range(cnt):
                    try:
                        item = parsed[k]
                        dx = float(item.get("dx", 0.0))
                        dy = float(item.get("dy", 0.0))
                        dz = float(item.get("dz", 0.0))
                    except Exception:
                        dx = dy = dz = 0.0
                    behaviors[i0 + k, :] = (dx, dy, dz)

            except Exception as e:
                print(f"[LLM parse error @ {i0}:{i0+cnt-1}] {e} — filling zeros.")

        return behaviors
            



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
                        + ", ".join([f"{key.capitalize()}={entry[key]:.3e}" for key in entry if key != "ite"])
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
        explanations = []

        with open(prompt_path, "r") as f:
            system_prompt = f.read()
        
        batch_results = self.call_llm_batch(
        particle_states=particle_states,
        history_states=history_states,
        batch_size=batch_size,
        system_prompt=system_prompt,
        max_tokens_per_particle=200,
        temperature=0.7,
        )    
        behaviors = self.parse_llm_batch_responses(
        batch_results=batch_results,
        num_particles=num_particles,
        )

        return behaviors
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


