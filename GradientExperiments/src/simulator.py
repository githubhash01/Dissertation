import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
import numpy as np
from jax import jacfwd  # Forward-mode autodiff

# TODO - use jacrev instead of jacfwd
#

# --- Auto-Differentiation ---
def make_step_fn(model, mjx_data):

    @jax.jit
    def step_fn(state):
        nq = model.nq
        qpos, qvel = state[:nq], state[nq:]
        dx = mjx_data.replace(qpos=qpos, qvel=qvel)
        dx_next = mjx.step(model, dx)
        next_state = jnp.concatenate([dx_next.qpos, dx_next.qvel])
        return next_state

    return step_fn

# --- Finite Differences ---
def make_step_fn_fd(mjx_model, mjx_data):
    epsilon = 1e-5

    # This function performs the actual step computation.
    @jax.jit
    def _step_fn(state):
        nq = mjx_model.nq
        qpos, qvel = state[:nq], state[nq:]
        dx = mjx_data.replace(qpos=qpos, qvel=qvel)
        dx_next = mjx.step(mjx_model, dx)
        next_state = jnp.concatenate([dx_next.qpos, dx_next.qvel])
        return next_state

    # Define the custom_vjp-decorated function.
    @jax.custom_vjp
    def step_fn(state):
        return _step_fn(state)

    # Forward pass: compute the output and return the input (or any auxiliary data)
    def step_fn_fwd(s):
        f_s = _step_fn(s)
        return f_s, s  # saving s for use in backward pass

    # Backward pass: given the saved s and the incoming cotangent,
    # compute the vector-Jacobian product using finite differences.
    def step_fn_bwd(s, cotangent):
        f_s = _step_fn(s)  # compute baseline output once
        grad = []
        for j in range(s.shape[0]):
            # Create a perturbation along the j-th coordinate.
            e_j = jnp.zeros_like(s).at[j].set(1.0)
            # Evaluate the function at the perturbed state.
            f_perturbed = _step_fn(s + epsilon * e_j)
            # Approximate the partial derivative for coordinate j.
            diff = (f_perturbed - f_s) / epsilon
            # Multiply by the cotangent to get the j-th component of the VJP.
            grad_j = jnp.vdot(cotangent, diff)
            grad.append(grad_j)
        grad = jnp.stack(grad)
        return (grad,)

    # Register the forward and backward functions with the custom_vjp mechanism.
    step_fn.defvjp(step_fn_fwd, step_fn_bwd)
    return step_fn

# --- Auto-Differentiation that takes mjx_data as input ----
# --- Only used for debugging purposes ---
def make_step_fn_dx(model, mjx_data):

    @jax.jit
    def step_fn(dx):
        # dx is your current mjx_data object (or a JAX-pytree with fields like qpos and qvel)
        dx_next = mjx.step(model, dx)
        return dx_next

    return step_fn

# --- Simulate the finger ---
def simulate_finger(mjx_data, num_steps, step_function):

    dx = mjx_data
    qpos = dx.qpos
    qvel = dx.qvel

    print(qpos, qvel)
    state = jnp.concatenate([qpos, qvel])
    print(state)
    states = [state]
    dxs = [dx]

    for _ in range(num_steps):

        dx_next = step_function(dx)
        #print(dx_next)
        dxs.append(dx_next)
        state_next = jnp.concatenate([dx_next.qpos, dx_next.qvel])
        states.append(state_next)

    return states

# --- Simulate the pusher ---
def simulate_pusher(mjx_data, num_steps, step_function):

    dx = mjx_data
    qpos = dx.qpos
    qvel = dx.qvel
    state = jnp.concatenate([qpos, qvel])
    states = [state]
    dxs = [dx]


    for _ in range(num_steps):

        dx_next = step_function(dx)
        dxs.append(dx_next)
        state_next = jnp.concatenate([dx_next.qpos, dx_next.qvel])
        states.append(state_next)

    return states

# --- Simulate the ball ---
def simulate(mjx_data, num_steps, step_function):
    state = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])
    states = [state]
    state_jacobians = []

    # define the backwards jacobian functio
    #jac_fn = jax.jit(jacfwd(step_function))  # Forward-mode is safer for loopsn
    jac_fn_rev = jax.jit(jax.jacrev(step_function))

    for _ in range(num_steps):
        # Compute Jacobian BEFORE stepping (gradient of NEXT state w.r.t. CURRENT state)
        #J_s = jac_fn(state)
        J_s = jac_fn_rev(state)

        state_jacobians.append(J_s)

        # Step forward
        state = step_function(state)
        states.append(np.array(state))

    states, state_jacobians = np.array(states), np.array(state_jacobians)
    return states, state_jacobians


def simulate_(mjx_data, num_steps, step_function):
    # Construct the initial state from the data.
    init_state = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])
    state_dim = init_state.shape[0]
    # Assume the Jacobian is square (state_dim x state_dim)
    jac_shape = (state_dim, state_dim)

    # Compile the reverse-mode Jacobian of the step function.
    jac_fn_rev = jax.jit(jax.jacrev(step_function))

    # Pre-allocate output arrays:
    # Trajectory: store num_steps+1 states (including initial state)
    traj = jnp.zeros((num_steps + 1, state_dim))
    # Jacobians: store num_steps Jacobians (one per step)
    jac_arr = jnp.zeros((num_steps, state_dim, state_dim))

    # Set the initial state as the first element in the trajectory.
    traj = traj.at[0].set(init_state)

    def body_fun(i, carry):
        """
        i: loop index (an integer)
        carry: tuple (current_state, traj, jac_arr)

        In each iteration:
          - Compute the Jacobian at the current state.
          - Take one simulation step.
          - Record the new state and Jacobian in our pre-allocated arrays.
        """
        state, traj, jac_arr = carry
        # Compute the Jacobian at the current state.
        J_s = jac_fn_rev(state)
        # Compute the next state.
        new_state = step_function(state)
        # Record the new state in the trajectory at position i+1.
        traj = traj.at[i + 1].set(new_state)
        # Record the computed Jacobian for this step at index i.
        jac_arr = jac_arr.at[i].set(J_s)
        # Return the updated carry.
        return (new_state, traj, jac_arr)

    # Ensure that num_steps is a concrete Python integer.
    num_steps = int(num_steps)
    # Use lax.fori_loop with static start (0) and stop (num_steps).
    final_carry = jax.lax.fori_loop(0, num_steps, body_fun, (init_state, traj, jac_arr))
    final_state, traj, jac_arr = final_carry
    return traj, jac_arr
