"""
Main file for doing analysis on the data generated from the experiments
"""

import os
import numpy as np
from GradientExperiments.src.visualiser import visualise_traj_generic
from GradientExperiments.src.experiments.one_bounce.config import mj_data, mj_model
import pandas as pd

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
    print_state_jacobian(jacobians[600], mj_model)

    print("Analysis completed")

if __name__ == "__main__":
    main()