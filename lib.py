import jax
import equinox as eqx

def linear_beta_schedule(num_steps, start=0.0001, end=0.02):
    return jax.numpy.linspace(start, end, num_steps)

def cosine_schedule(num_timesteps, s=0.008):
    def f(t):
        return jax.numpy.cos((t / num_timesteps + s) / (1 + s) * 0.5 * jax.numpy.pi) ** 2
    x = jax.numpy.linspace(0, num_timesteps, num_timesteps + 1)
    alphas_cumprod = f(x) / f(jax.numpy.array([0]))
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas

def generate_pyramid_scheduling_matrix(sequence_length: int, uncertainty_scale: float, sampling_timesteps: int):
    height = sampling_timesteps + jax.numpy.round((sequence_length - 1) * uncertainty_scale).astype(jax.numpy.int32) + 1
    scheduling_matrix = jax.numpy.zeros((height, sequence_length)).astype(jax.numpy.int32)
    def fill_matrix(m, t):
        return sampling_timesteps + (t * uncertainty_scale).astype(jax.numpy.int32) - m
    scheduling_matrix = jax.vmap(lambda m: jax.vmap(lambda t: fill_matrix(m, t))(jax.numpy.arange(sequence_length)))(jax.numpy.arange(height))
    return jax.numpy.clip(scheduling_matrix, 0, sampling_timesteps)

def generate_trapezoid_scheduling_matrix(sequence_length: int, uncertainty_scale: float, sampling_timesteps: int):
    height = sampling_timesteps + jax.numpy.round((sequence_length + 1) // 2 * uncertainty_scale).astype(jax.numpy.int32)
    scheduling_matrix = jax.numpy.zeros((height, sequence_length)).astype(jax.numpy.int32)
    def fill_matrix(m, t):
        return sampling_timesteps + (t * uncertainty_scale).astype(jax.numpy.int32) - m
    half_fill_matrix = jax.vmap(lambda m: jax.vmap(lambda t: fill_matrix(m, t))(jax.numpy.arange((sequence_length + 1) // 2)))(jax.numpy.arange(height))
    scheduling_matrix = scheduling_matrix.at[:, :(sequence_length + 1) // 2].set(half_fill_matrix)
    scheduling_matrix = scheduling_matrix.at[:, -(sequence_length + 1) // 2:].set(half_fill_matrix)
    return jax.numpy.clip(scheduling_matrix, 0, sampling_timesteps)

def generate_full_scheduling_matrix(sequence_length: int, sampling_timesteps: int):
    return jax.numpy.arange(sampling_timesteps, -1, -1)[:, None].repeat(sequence_length, axis=1)

def predict_v(schedule, t, x_0, noise):
    betas = schedule
    alphas = 1 - betas
    alpha_hats = jax.numpy.cumprod(alphas)
    alpha_hats_prev = jax.numpy.concatenate([jax.numpy.array([1.0]), alpha_hats])
    return jax.numpy.sqrt(alpha_hats_prev[t + 1]) * noise - jax.numpy.sqrt(1 - alpha_hats_prev[t + 1]) * x_0

def predict_x_t(schedule, t, x_0, noise):
    betas = schedule
    alphas = 1 - betas
    alpha_hats = jax.numpy.cumprod(alphas)
    alpha_hats_prev = jax.numpy.concatenate([jax.numpy.array([1.0]), alpha_hats])
    return jax.numpy.sqrt(alpha_hats_prev[t + 1]) * x_0 + jax.numpy.sqrt(1 - alpha_hats_prev[t + 1]) * noise

def predict_x_0(schedule, t, x_t, v):
    betas = schedule
    alphas = 1 - betas
    alpha_hats = jax.numpy.cumprod(alphas)
    alpha_hats_prev = jax.numpy.concatenate([jax.numpy.array([1.0]), alpha_hats])
    return jax.numpy.sqrt(alpha_hats_prev[t + 1]) * x_t - jax.numpy.sqrt(1 - alpha_hats_prev[t + 1]) * v

def sample(noise_schedule, diffusion_schedule, temperature, model_state, x_list, model_fn, rng):
    """
    Sampler for sequence models that were trained with diffusion forcing.
    
    sequence models are models that accept arbitrary lengths of inputs and product arbitary lengths of inputs.
    diffusion forcing is a training strategy for sequence models. it randomly adds noise per timestep in the sequence.

    Throughout this function and in docs, the sampling timesteps are governed 


    noise_schedule: jax.Array shape: (S,). betas that tell you how much noise to apply at each diffusion timestep
    diffusion_schedule: jax.Array that specifies the diffusion schedule. use the above functions to generate the list.
    temperature: float
    model_state: whatever you want. this should be a pytree. I use this to pass in my model parameters and stuff like padding
    x_list: inputs
    model_fn: callable that takes the 
    rng: jax.Array generated from jax.random.key(0)
    output: tuple. list of denoised values in time

    Take a diffusion schedule like the following:
    >>> diffusion_utils.generate_pyramid_scheduling_matrix(2, 2, 2)
    Array([[2, 2],
           [1, 2],
           [0, 2],
           [0, 1],
           [0, 0]], dtype=int32)
    This tells you that we are going to treat x_list as if it were completely random. We can cleave this if we know that the first element of x_list has already
    been denoised or was given to us. The following schedule array would have the first element of x_list be treated as if it were already denoised:
    [[0, 2],
     [0, 1],
     [0, 0]]
    
    Onto the internals of this function. We have a scan function which is responsible for the looping. This is pretty simple when you peel back the irritating jax
    looping syntax. We have a for loop over the diffusion schedule. We will carry the currently predicted completely denoised list and every timestep we update
    this denoised list.
    """
    rng, key = jax.random.split(rng)
    dynamic_model_state, static_model_state = eqx.partition(model_state, eqx.is_array)
    def body(carry, diffusion_step_list):
        x_0_list, dynamic_model_state, rng = carry
        rng, key = jax.random.split(rng)
        noise_list = [diffusion_step > 0 * temperature * jax.random.normal(key, x_0.shape) for (diffusion_step, key, x_0) in zip(diffusion_step_list, jax.random.split(key, len(x_0_list)), x_0_list)]
        x_t_list = [predict_x_t(noise_schedule, diffusion_step, x_0, noise) for (diffusion_step, x_0, noise) in zip(diffusion_step_list, x_0_list, noise_list)]
        model_state = eqx.combine(dynamic_model_state, static_model_state)
        v_list, model_state = model_fn(diffusion_step_list, x_t_list, model_state)
        dynamic_model_state, _ = eqx.partition(model_state, eqx.is_array)
        rng, key = jax.random.split(rng)
        x_0_list = [predict_x_0(noise_schedule, diffusion_step, x_t, v) for (diffusion_step, x_t, v) in zip(diffusion_step_list, x_t_list, v_list)]
        rng, key = jax.random.split(rng)
        return (x_0_list, dynamic_model_state, key), x_t_list
    rng, key = jax.random.split(rng)
    (x_0_list, _, _), x_0_progress_list = jax.lax.scan(
        body,
        (x_list, dynamic_model_state, key),
        diffusion_schedule
    )
    return x_0_list, x_0_progress_list

import unittest
class TestSample(unittest.TestCase):
    def test_sample(self):
        sampling_steps = 3
        sequence_length = 2
        temperature = 1.0
        dimension = 10
        model_state = None
        noise_schedule = cosine_schedule(sampling_steps)
        diffusion_schedule = generate_pyramid_scheduling_matrix(sequence_length, sampling_steps, sampling_steps)
        model_fn = lambda _, x_list, model_state: (x_list, model_state)
        x_list = [jax.random.normal(jax.random.key(0), (dimension)) for _ in range(sequence_length)]
        result, _ = sample(noise_schedule, diffusion_schedule, temperature, model_state, x_list, model_fn, jax.random.key(0))
        for res in result:
            assert res.shape == (dimension,), f"Expected shape ({dimension},), but got {res.shape}"
        assert len(result) == sequence_length, f"Expected sequence length {sequence_length}, but got {len(result)}"
if __name__ == '__main__':
    unittest.main()
