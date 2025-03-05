"""
Main file for doing analysis on the data generated from the experiments
"""

import os
import numpy as np
from GradientExperiments.src.visualiser import visualise_traj_generic
from GradientExperiments.src.experiments.one_bounce.config import mj_data, mj_model
import pandas as pd
import matplotlib.pyplot as plt

def print_state_jacobian(jacobian_state, mujoco_model):

    nq = mujoco_model.nq  # expected to be 7
    nv = mujoco_model.nv  # expected to be 6
    # Define labels for qpos and qvel
    #qpos_labels = ['p_x', 'p_y', 'p_z', 'q_w', 'q_x', 'q_y', 'q_z']
    #qvel_labels = ['v_x', 'v_y', 'v_z', 'ω_x', 'ω_y', 'ω_z']

    # Extract blocks from the full Jacobian
    # dq_next/dq: top-left block (nq x nq)
    dq_dq = jacobian_state[:nq, :nq]
    # dq_next/dv: top-right block (nq x nv)
    dq_dv = jacobian_state[:nq, nq:]
    # dv_next/dq: bottom-left block (nv x nq)
    dv_dq = jacobian_state[nq:, :nq]
    # dv_next/dv: bottom-right block (nv x nv)
    dv_dv = jacobian_state[nq:, nq:]

    # Create DataFrames for better formatting in terminal output
    df_dq_dq = pd.DataFrame(dq_dq)
    df_dq_dv = pd.DataFrame(dq_dv)
    df_dv_dq = pd.DataFrame(dv_dq)
    df_dv_dv = pd.DataFrame(dv_dv)

    # Print the blocks with headers
    print("Jacobian Block: dq_next/dq (Position w.r.t. Position)")
    print(df_dq_dq)
    print("\nJacobian Block: dq_next/dv (Position w.r.t. Velocity)")
    print(df_dq_dv)
    print("\nJacobian Block: dv_next/dq (Velocity w.r.t. Position)")
    print(df_dv_dq)
    print("\nJacobian Block: dv_next/dv (Velocity w.r.t. Velocity)")
    print(df_dv_dv)


def build_ground_truth_jacobian(dt):
    """
    Construct a 13x13 Jacobian for a free joint in 3D,
    under the simplified rule:
      p' = p + dt * v
      q' = q   (ignore orientation update)
      v' = v
      w' = w
    Hence:
      dq'/dq = I7,  dq'/dv = [ dt*I3 ; 0 ],
      dv'/dq = 0,   dv'/dv = I6.
    """
    nq, nv = 7, 6  # free joint in 3D
    J_gt = np.zeros((nq + nv, nq + nv))

    # (1) dq'/dq -> top-left block: 7x7 identity
    J_gt[:nq, :nq] = np.eye(nq)

    # (2) dv'/dv -> bottom-right block: 6x6 identity
    J_gt[nq:, nq:] = np.eye(nv)

    # (3) dq'/dv -> top-right block:
    #   - for the first 3 DOFs of qpos (p_x, p_y, p_z),
    #     partial wrt. the first 3 DOFs of velocity (v_x, v_y, v_z) is dt.
    #     That corresponds to rows [0..2], columns [0..2] *within the sub-block*.
    #   - The sub-block itself sits at rows [0..6], columns [7..12] in the full matrix.
    #   - So for i in [0,1,2], j in [0,1,2]:
    for i in range(3):
        J_gt[i, nq + i] = dt

    # (4) dv'/dq -> bottom-left block: 6x7 zero (already zero from initialization)

    return J_gt


def plot_jacobian_difference_across_time(
        jacobians,
        ground_truth,
        boundaries,  # tuple like (start_collision, end_collision)
        title="Jacobian difference across time"
):
    """
    Plots the Jacobian difference in three separate subplots:
    - Pre-collision
    - Collision
    - Post-collision

    Parameters:
    - jacobians: (T, nq+nv, nq+nv) array of computed Jacobians at each time step.
    - ground_truth: (nq+nv, nq+nv) analytical/expected Jacobian.
    - boundaries: (start_collision, end_collision) defining when collision occurs.
    - title: Overall title for the figure.
    """

    jacobians = np.array(jacobians)
    T = jacobians.shape[0]
    time_steps = np.arange(T)

    # Compute Frobenius norm error at each time step
    errors = np.linalg.norm(jacobians - ground_truth, ord='fro', axis=(1, 2))

    # Unpack boundaries
    start_collision, end_collision = boundaries

    # Split data into three phases
    pre_mask = time_steps < start_collision
    col_mask = (time_steps >= start_collision) & (time_steps < end_collision)
    post_mask = time_steps >= end_collision

    # Set consistent y-limits (optional)
    y_min, y_max = np.min(errors), np.max(errors)

    # Create figure and 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # Plot Pre-Collision
    axs[0].plot(time_steps[pre_mask], errors[pre_mask], 'bo-')
    axs[0].set_title("Pre-Collision")
    axs[0].set_ylabel("Jacobian Difference (Fro. norm)")
    axs[0].grid(True)
    axs[0].set_ylim(y_min, 5)

    # Plot Collision
    axs[1].plot(time_steps[col_mask], errors[col_mask], 'ro-')
    axs[1].set_title("During Collision")
    axs[1].set_ylabel("Jacobian Difference (Fro. norm)")
    axs[1].grid(True)
    axs[1].set_ylim(y_min, y_max)

    # Plot Post-Collision
    axs[2].plot(time_steps[post_mask], errors[post_mask], 'go-')
    axs[2].set_title("Post-Collision")
    axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("Jacobian Difference (Fro. norm)")
    axs[2].grid(True)
    axs[2].set_ylim(y_min, 5)

    # Set the overall figure title
    plt.suptitle(title)
    plt.show()

    # Print max error stats for debugging
    print(f"Max error pre-collision:  {np.max(errors[pre_mask]):.6e}")
    print(f"Max error during collision:  {np.max(errors[col_mask]):.6e}")
    print(f"Max error post-collision: {np.max(errors[post_mask]):.6e}")


def main():

    print("Running analysis on the data ...")

    # Load the data
    stored_data_directory = "/Users/hashim/Desktop/Dissertation/GradientExperiments/src/experiments/one_bounce/stored_data"
    #stored_data_directory ="/Users/hashim/Desktop/Dissertation/GradientExperiments/src/experiments/two_bounce/stored_data"

    fd = "jacobians_fd.npy"
    autodiff = "jacobians_autodiff.npy"
    implicit = "jacobians_implicit.npy"

    #states = np.load(os.path.join(stored_data_directory, 'states_fd.npy'))
    jacobians = np.load(os.path.join(stored_data_directory, fd))

    # visualise the trajectory using the states data
    print("Visualising the trajectory ...")
    #visualise_traj_generic(states, mj_data, mj_model)

    # Perform analysis on the data
    print_state_jacobian(jacobians[300], mj_model)

    # Build up the ground truth Jacobian
    J_gt = build_ground_truth_jacobian(dt=0.01)
    # use the ground truth jacobian to plot the difference between the jacobians
    plot_jacobian_difference_across_time(jacobians, J_gt, boundaries=[450, 550], title="Jacobian difference across time")


    print("Analysis completed")

if __name__ == "__main__":
    main()