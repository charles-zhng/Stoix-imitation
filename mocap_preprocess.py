"""Preprocessing for embedding motion capture/dannce data."""
import dm_control
import h5py
from dm_control.locomotion.walkers import rodent
from dm_control.locomotion.walkers import rescale
from dm_control.utils import transformations as tr
from dm_control import mjcf
import pickle
import numpy as np
# from scipy.io import loadmat
from typing import Text, List, Tuple, Dict, Union
import jax
from jax import numpy as jp
from flax import struct
from typing import Any

# 13 features
@struct.dataclass
class ReferenceClip():
    angular_velocity: jp.ndarray
    appendages: jp.ndarray
    body_positions: jp.ndarray
    body_quaternions: jp.ndarray
    center_of_mass: jp.ndarray
    end_effectors: jp.ndarray
    joints: jp.ndarray
    joints_velocity: jp.ndarray
    markers: jp.ndarray
    position: jp.ndarray
    quaternion: jp.ndarray
    scaling: jp.ndarray
    velocity: jp.ndarray
    
def save_dataclass_pickle(pickle_path, mocap_features):
    data = ReferenceClip(**mocap_features)
    data = jax.tree_map(lambda x: jp.array(x), data)
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)
    return pickle_path


def save_features(file: h5py.File, mocap_features: Dict, clip_name: Text):
    """Save features to hdf5 dataset

    Args:
        file (h5py.File): Hdf5 dataset
        mocap_features (Dict): Features extracted through rollout
        clip_name (Text): Name of the clip stored in the hdf5 dataset.
    """
    clip_group = file.create_group(clip_name)
    n_steps = len(mocap_features["center_of_mass"])
    clip_group.attrs["num_steps"] = n_steps
    clip_group.attrs["dt"] = 0.02
    file.create_group("/" + clip_name + "/walkers")
    file.create_group("/" + clip_name + "/props")
    walker_group = file.create_group("/" + clip_name + "/walkers/walker_0")
    for k, v in mocap_features.items():
        if len(np.array(v).shape) == 3:
            v = np.transpose(v, (1, 2, 0))
            # print(v.shape)
            walker_group[k] = np.reshape(np.array(v), (-1, n_steps))
        elif len(np.array(v).shape) == 2:
            v = np.swapaxes(v, 0, 1)
            walker_group[k] = v
        else:
            walker_group[k] = v