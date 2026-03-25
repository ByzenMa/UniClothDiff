import argparse
import os
import numpy as np
import trimesh
import h5py
from matplotlib import pyplot as plt
import open3d as o3d
import numpy as np


def ply_to_obj_open3d(ply_filepath, obj_filepath):
    # Read the mesh from the PLY file
    # Open3D automatically detects the file type from the extension
    mesh = o3d.io.read_triangle_mesh(ply_filepath)

    if not mesh.has_vertices():
        print("Error: No vertices found in the PLY file.")
        return

    # Write the mesh to the OBJ file
    # Open3D automatically handles the conversion to the target format
    o3d.io.write_triangle_mesh(obj_filepath, mesh, write_ascii=True)
    print(f"Successfully converted {ply_filepath} to {obj_filepath}")


# Loading utilities
def load_objects(obj_root, select_points=True, select_num=100):
    object_names = ['juice', 'liquid_soap', 'milk', 'salt']
    all_models = {}
    for obj_name in object_names:
        obj_path = os.path.join(obj_root, '{}_model'.format(obj_name),
                                '{}_model.ply'.format(obj_name))
        mesh = trimesh.load(obj_path)

        if select_points:
            pcd = o3d.io.read_point_cloud(obj_path)
            points = np.asarray(pcd.points)
            random_indices = np.random.choice(points.shape[0], size=select_num, replace=False)
            new_pcd = pcd.select_by_index(random_indices)
            saved_path = os.path.join(obj_root, '{}_model'.format(obj_name), '{}_selected_{}_points.ply'.format(obj_name, select_num))
            o3d.io.write_point_cloud(saved_path, new_pcd)
            mesh = trimesh.load(saved_path)

        all_models[obj_name] = {
            'verts': np.array(mesh.vertices),
            # 'faces': np.array(mesh.faces)
        }
    return all_models


def get_skeleton(sample, skel_root):
    skeleton_path = os.path.join(skel_root, sample['subject'],
                                 sample['action_name'], sample['seq_idx'],
                                 'skeleton.txt')
    skeleton_vals = np.loadtxt(skeleton_path)
    skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21,
                                            -1)[sample['frame_idx']]
    return skeleton


def get_obj_transform(sample, obj_root):
    seq_path = os.path.join(obj_root, sample['subject'], sample['action_name'],
                            sample['seq_idx'], 'object_pose.txt')

    with open(seq_path, 'r') as seq_f:
        raw_lines = seq_f.readlines()
    raw_line = raw_lines[sample['frame_idx']]
    line = raw_line.strip().split(' ')
    trans_matrix = np.array(line[1:]).astype(np.float32)
    trans_matrix = trans_matrix.reshape(4, 4).transpose()
    return trans_matrix


# Display utilities
def visualize_joints_2d(ax, joints, joint_idxs=True, links=None, alpha=1):
    """Draw 2d skeleton on matplotlib axis"""
    if links is None:
        links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    ax.scatter(x, y, 1, 'r')

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw2djoints(ax, joints, links, alpha=alpha)


def _draw2djoints(ax, annots, links, alpha=1):
    """Draw segments, one color per link"""
    colors = ['r', 'm', 'b', 'c', 'g']

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha)


def _draw2dseg(ax, annot, idx1, idx2, c='r', alpha=1):
    """Draw segment of given color"""
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]],
        c=c,
        alpha=alpha)


def get_verts_actions(sample):
    reorder_idx = np.array([
        0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19,
        20
    ])
    root = sample['root']
    cam_extr = np.array(
        [[0.999988496304, -0.00468848412856, 0.000982563360594,
          25.7], [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
         [-0.000969709653873, 0.00274303671904, 0.99999576807,
          3.902], [0, 0, 0, 1]])
    skeleton_root = os.path.join(root, 'Hand_pose_annotation_v1')
    obj_root = os.path.join(root, 'Object_models')
    obj_trans_root = os.path.join(root, 'Object_6D_pose_annotation_v1_1')
    skel = get_skeleton(sample, skeleton_root)[reorder_idx]
    # Load object mesh
    object_infos = load_objects(obj_root, select_points=sample['select_points'], select_num=sample['select_num'])

    # Load object transform
    obj_trans = get_obj_transform(sample, obj_trans_root)

    # Get object vertices
    verts = object_infos[sample['object']]['verts'] * 1000

    # Apply transform to object
    hom_verts = np.concatenate(
        [verts, np.ones([verts.shape[0], 1])], axis=1)
    verts_trans = obj_trans.dot(hom_verts.T).T

    # Apply camera extrinsic to objec
    verts_camcoords = cam_extr.dot(verts_trans.transpose()).transpose()[:, :3]
    # Apply camera extrinsic to hand skeleton
    skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
    skel_camcoords = cam_extr.dot(skel_hom.transpose()).transpose()[:, :3].astype(np.float32)

    return verts_camcoords, skel_camcoords


def get_data_info_list(sample):
    frame_num = sample['frame_num']
    data_dict = {}
    for idx in range(frame_num):
        sample['frame_idx'] = idx
        # print("processing sample {}".format(sample))
        root = sample['root']
        obj_trans_root = os.path.join(root, 'Object_6D_pose_annotation_v1_1')
        seq_path = os.path.join(obj_trans_root, sample['subject'], sample['action_name'],
                                sample['seq_idx'], 'object_pose.txt')
        if not os.path.exists(seq_path):
            continue

        verts, actions = get_verts_actions(sample)
        data_dict[idx] = (verts, actions)
    return data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=False, default='F-PHAB/')
    parser.add_argument('--output', type=str, required=False, default='HandPose/')
    parser.add_argument('--num_prev_frames', type=int, required=False, default=3)
    parser.add_argument('--num_next_frames', type=int, required=False, default=1)
    parser.add_argument('--select_points', type=bool, required=False, default=True)
    parser.add_argument('--select_num', type=int, required=False, default=10)
    parser.add_argument('--obj', type=list, default=['liquid_soap'])
    args = parser.parse_args()

    # get subinfos
    sub_info_path = os.path.join(args.root, 'Subjects_info')
    sub_files = os.listdir(sub_info_path)
    data_samples = []
    for file in sub_files:
        sub_name = file.split('_info')[0]
        sub_path = os.path.join(sub_info_path, file)
        with open(sub_path, 'r') as f:
            for line in f:
                for kw in args.obj:
                    if kw in line:
                        dir_name, part_id, _, frames_num = line.strip('\n').split(' ')
                        sample = {
                            'root': args.root,
                            'subject': sub_name,
                            'action_name': dir_name,
                            'seq_idx': part_id,
                            'object': kw,
                            'frame_num': int(frames_num),
                            'select_points': args.select_points,
                            'select_num': args.select_num,
                        }
                        data_samples.append(sample)

    # get all frame info from one video
    for sample in data_samples:

        sub_dir = sample['object']
        # make output directory
        output_path = os.path.join(args.output, sub_dir)
        os.makedirs(output_path, exist_ok=True)

        data_info_list = get_data_info_list(sample)
        frame_num = sample['frame_num']
        num_prev_frames = args.num_prev_frames
        num_next_frames = args.num_next_frames
        if frame_num != len(data_info_list):
            continue
        for idx in range(num_prev_frames, frame_num-num_next_frames+1):
            q_prev, q_next, action = [], [], []
            for i in range(num_prev_frames):
                v, a = data_info_list[idx-num_prev_frames+i]
                q_prev.append(v)
            for i in range(num_next_frames):
                v, a = data_info_list[idx+i]
                q_next.append(v)
                action.append(a)
            q_prev = np.array(q_prev)
            q_next = np.array(q_next)
            action = np.array(action)
            h5_name = "{}_{}_{}_{}".format(sample['subject'], sample['action_name'], sample['seq_idx'], idx)
            h5_file_path = os.path.join(output_path, "{}.hdf5".format(h5_name))
            with h5py.File(h5_file_path, 'w') as f:
                f.create_dataset('q_prev', data=q_prev)
                f.create_dataset('q_next', data=q_next)
                f.create_dataset('action', data=action)

        print("create data from sample {} done".format(sample))




