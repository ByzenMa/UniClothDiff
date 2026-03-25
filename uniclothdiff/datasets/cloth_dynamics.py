import os
import numpy as np

import torch
from torch.utils.data import Dataset

import torch.nn.functional as F
import math
import h5py
from uniclothdiff.utils import calibur
from uniclothdiff.registry import DATASETS


def read_q_bounds(bounds_path):
    with open(bounds_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # new format:
    # min_q x y z
    # max_q x y z
    # max_delta_q v
    if lines[0].startswith("min_q"):
        min_q = np.array([float(v) for v in lines[0].split()[1:4]], dtype=np.float32)
        max_q = np.array([float(v) for v in lines[1].split()[1:4]], dtype=np.float32)
        max_delta_q = float(lines[2].split()[1]) if len(lines) > 2 else None
        return min_q, max_q, max_delta_q

    # backward compatibility: 2x3 numeric matrix.
    bounds = np.loadtxt(bounds_path, dtype=np.float32)
    if bounds.shape != (2, 3):
        raise ValueError(f"Invalid bounds shape: {bounds.shape}, expected (2, 3)")
    return bounds[0], bounds[1], None


@DATASETS.register_module()
class ClothDynamicsDataset(Dataset):

    def __init__(
        self, 
        data_dir: str, 
        min_q=None,
        max_q=None,
        max_delta_q=0.05,
        mode='train',
        num_prev_frames=3,
        num_next_frames=1,
        action_mode='delta_ee_pose',
        max_delta_action=0.02,
        num_patches=100,
        patch_size=25,
        view_matrix=None,
        bounds_file='q_bounds.txt',
    ):

        self.data_dir = data_dir
        self.data_files = sorted([
            file_name for file_name in os.listdir(self.data_dir)
            if file_name.endswith('.hdf5') or file_name.endswith('.h5')
        ])
        self.num_prev_frames = num_prev_frames
        self.num_next_frames = num_next_frames
        self.max_delta_action = max_delta_action
        
        assert action_mode in ['ee_pose', 'delta_ee_pose']
        self.action_mode = action_mode
        num_data_files = len(self.data_files)
        
        if mode == 'train':
            self.data_files = self.data_files[:int(num_data_files * 0.9)]                
        else:
            self.data_files = self.data_files[int(num_data_files * 0.9):]
        self.mode = mode
        
        self.num_samples = len(self.data_files)
        
        if min_q is None or max_q is None:
            bounds_path = os.path.join(self.data_dir, bounds_file)
            if os.path.exists(bounds_path):
                min_q, max_q, file_max_delta_q = read_q_bounds(bounds_path)
                if file_max_delta_q is not None:
                    max_delta_q = file_max_delta_q
            else:
                min_q = [-1.0, -1.0, -1.0]
                max_q = [1.0, 1.0, 1.0]
        self.min_q = torch.tensor(np.array(min_q), dtype=torch.float32)
        self.max_q = torch.tensor(np.array(max_q), dtype=torch.float32)
        self.max_delta_q = max_delta_q

        self.view_matrix_ros = view_matrix if view_matrix is not None else torch.eye(4)
        self.view_matrix_gl = calibur.convert_pose(self.view_matrix_ros, calibur.CC.ROS, calibur.CC.OpenGL)

        self.projection_matrix = self.projection()
        
        self.num_patches = num_patches
        self.patch_size = patch_size

    def __len__(self):
        return self.num_samples
    
    def normalize_q(self, q):
        normalized_points = 2.0 * (q - self.min_q) / (self.max_q - self.min_q) - 1.0
        return normalized_points
    
    def denormalize_xyz(self, norm_q):
        # Assume min_xyz and max_xyz are already defined in the class
        min_vals = self.min_q
        max_vals = self.max_q
        
        # Calculate the original coordinates from normalized points
        q = (norm_q + 1.0) * ((max_vals - min_vals) / 2.0) + min_vals
    
        return q
    
    def normalize_delta_q(self, q):
        min_delta_q = -1.0 * self.max_delta_q
        max_delta_q = self.max_delta_q

        norm_delta_q = 2.0 * (q - min_delta_q) / (max_delta_q - min_delta_q) - 1.0
        return norm_delta_q
    
    def normalize_action(self, action):
        min_delta_action = -1.0 * self.max_delta_action
        max_delta_action = self.max_delta_action
        
        norm_delta_action = 2.0 * (action - min_delta_action) / (max_delta_action - min_delta_action) - 1.0
        return norm_delta_action

    def vertices_to_2d(self, q, resolution):
        width, height = resolution
        seq_len = q.shape[0]
        if q.shape[1] != width * height:
            raise ValueError("Number of vertices does not match resolution.")
        
        image_array = q.reshape((seq_len, height, width, 3))
        
        return image_array
    
    # Define fixed camera projection and modelview matrices
    def projection(self, fovy=0.75943, aspect=1.767, near=0.1, far=100):
        f = 1.0 / np.tan(fovy / 2.0)
        proj = np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0,0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1 , 0]
        ], dtype=np.float32)
        return torch.tensor(proj, dtype=torch.float32)
    
    
    def __getitem__(self, idx):
        file_name = self.data_files[idx]
        file_path = os.path.join(self.data_dir, file_name)
        
        try:
            with h5py.File(file_path, 'r') as raw_data:
                point_idx = raw_data['point_index'][()]
                q_prev = raw_data['q_prev'][:]
                q_next = raw_data['q_next'][:]
                q_prev = torch.tensor(q_prev, dtype=torch.float32)
                q_next = torch.tensor(q_next, dtype=torch.float32)
                
                if q_next.ndim == 2:
                    q_next = q_next[None, ...]
                else:
                    q_next = q_next
                
                q_prev_homo = torch.cat((q_prev, torch.ones(q_prev.shape[0], q_prev.shape[1], 1)), dim=-1)
                q_next_homo = torch.cat((q_next, torch.ones(q_next.shape[0], q_next.shape[1], 1)), dim=-1)
                
                q_prev_camera = q_prev_homo @ self.view_matrix_ros.t()
                q_next_camera = q_next_homo @ self.view_matrix_ros.t()
                
                q_prev = q_prev_camera[:, :, :3]
                q_next = q_next_camera[:, :, :3]
                
                # get action
                action = torch.tensor(raw_data['action'][:], dtype=torch.float32)
                if action.ndim == 1:
                    action = action[None]
                if action.shape[0] != q_next.shape[0]:
                    assert point_idx < 0
                    action = action.repeat(q_next.shape[0], 1)
                action_homo = torch.cat([action, torch.ones(action.shape[0], 1)], dim=-1)
                action_camera = action_homo @ self.view_matrix_ros.t()
                action = action_camera[:, :3]
                abs_action = action.clone()
                # get mask
                q_mask = torch.zeros((q_prev.shape[1], 1), dtype=torch.float32)
                
                if point_idx >= 0:
                    q_mask[point_idx, :] = 1.0

                num_next_frame = q_next.shape[0]
                        
                if num_next_frame == 1:
                    q_delta = q_next - q_prev[-1:]
                    if self.action_mode == "delta_ee_pose":
                        action = action - q_prev[-1:, point_idx, :]
                else:
                    # delta with respect to last frame
                    prev_frames = torch.cat([q_prev[-1:], q_next[:-1]], dim=0)
                    q_delta = q_next - prev_frames
                    if self.action_mode == "delta_ee_pose":
                        prev_action = torch.cat([q_prev[-1:, point_idx, :], action[:-1,  ...]], dim=0)
                        action = action - prev_action
                
                q_delta = self.normalize_delta_q(q_delta)
                if self.action_mode == "delta_ee_pose":
                    action = self.normalize_action(action)
                         
                if self.mode == 'test':
                    return {
                        "q_prev": q_prev,
                        "q_next": q_next,
                        "q_delta": q_delta,
                        "action": action,
                        "point_index": torch.tensor(point_idx),
                        "mask": q_mask[None, ...],
                        "abs_action": abs_action
                    }
                return {
                    "q_prev": q_prev,
                    "q_next": q_next,
                    "q_delta": q_delta,
                    "action": action,
                    "mask": q_mask[None, ...],
                    "abs_action": abs_action
                }
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None
