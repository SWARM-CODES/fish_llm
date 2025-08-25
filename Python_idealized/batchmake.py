def estimate_batch_size(prompt_file, token_limit=8000, system_token_estimate=100, particle_token_estimate=500, batch_size=None):
    """
    Estimate the maximum number of particles that can fit in a batch based on token limits.
    If a batch_size is provided and exceeds the token limit, the program stops with an error.
    """
    # Read the base prompt from the file
    with open(prompt_file, 'r') as f:
        base_prompt = f.read()
    
    # Approximation of token count for the base prompt
    base_prompt_tokens = len(base_prompt.split())
    
    # Calculate available tokens after accounting for system and prompt tokens
    available_tokens = token_limit - system_token_estimate - base_prompt_tokens
    
    # Calculate the maximum particles that fit within the token limit
    max_particles = available_tokens // particle_token_estimate
    
    # If batch_size is provided, validate it
    if batch_size:
        if batch_size > max_particles:
            raise ValueError(
                f"Provided batch_size ({batch_size}) exceeds the token limit! "
                f"The maximum allowed batch size is {max_particles}. "
                f"Please adjust the batch_size."
            )
        else:
            print(f"Provided batch_size ({batch_size}) is within the token limit.")
            return batch_size
    else:
        # Return the maximum particles if no batch_size is provided
        return max_particles


def divide_particles_into_batches(particles, max_particles_per_batch):
    """
    Sequentially divide particles into batches based on the maximum particles allowed per batch.
    """
    return [particles[i:i + max_particles_per_batch] for i in range(0, len(particles), max_particles_per_batch)]

