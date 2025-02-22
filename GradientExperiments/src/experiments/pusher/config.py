from dataclasses import dataclass
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import os

# Get the absolute path of the current file's directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the stored_data directory relative to this file.
stored_data_directory = os.path.join(BASE_DIR, "stored_data")

# --- Configuration ---
@dataclass
class Config:
    simulation_time: float = 10.0
    steps: int = 1000

    # create init pos of length 11
    init_pos = jnp.array([
        0,  # base x
        0,  # base y
        0,  # base z
        1.0, 0.0, 0.0, 0.0,  # base orientation as quaternion (w,x,y,z)
        1.0, 1.0, 1.0, 1.0  # arm joint angles (4 values)
    ])
    init_vel = jnp.array([2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ctrl_input = jnp.array([0.0, 0.0, 0.0])

xml_path = "/Users/hashim/Desktop/Dissertation/GradientExperiments/src/experiments/pusher/pusher.xml"
mj_model = mujoco.MjModel.from_xml_path(filename=xml_path)
mj_data = mujoco.MjData(mj_model)
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

config = Config()

# --- Initialize state ---

mjx_data = mjx_data.replace(
    qpos=config.init_pos,
    qvel=config.init_vel
    # in this case the qvel is also nv dimensional (derivatives of qpos)
    #qvel=jnp.array([config.init_vel[0], config.init_vel[1], config.init_vel[2], 0.0, 0.0, 0.0, 0.0])
)