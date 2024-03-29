import json
import ast
from tqdm import tqdm
import numpy as np
import os

def process_pmi_samples(keypoints_path):
    """
    Process samples from the PMI-GMA dataset, as obtained during this implementation.

    Parameters:
    keypoints_path (string): Path to the origin folder of pose data.

    Returns:
    (list): A list of Numpy arrays containing pose data.
    """
    samples_coords = []
    files = os.listdir(keypoints_path)
    for joint_list_name in tqdm(files, desc='Processing'):
        file_path = os.path.join(keypoints_path, joint_list_name)
        with open(file_path) as joint_list:
            joint_list = np.array([ast.literal_eval(coord) for coord in joint_list.readlines()])
            samples_coords.append((joint_list_name.strip('.txt'), joint_list))
    return samples_coords

def read_from_openpose(file_list, custom_prefix, samples, keypoints_path):
    """
    This function should read all files from OpenPose, given that they are on a single folder.

    Parameters:
    file_list (list): List containing all name files outputted from OpenPose.
    
    custom_prefix (string): The string prefixed to the OpenPose standard "Export_()_()_keypoints.json"
    
    samples (list): List containing the desired samples IDs/names.
    
    keypoints_path (string): The path to the folder containing the keypoints files.

    Returns:
    (list): A list of Numpy arrays shaped to contain 25 joints and 3 features (x,y,confidence score).
    """
    sample_coords = []
    sample = 0

    total_iter = len(file_list)
    curr_iter = 1
    progress_bar = tqdm(total=total_iter, desc='Processing')

    while(curr_iter < total_iter):
        coords_for_this_infant = []
        for i in file_list[curr_iter-1:]:
            if i.startswith(custom_prefix.format(samples[sample])):
                with open(os.path.join(keypoints_path, i), "r") as file:
                    data = json.load(file)
                    coords = np.array((data['people'][0]['pose_keypoints_2d'] ))
                    coords = coords.reshape(25,3)
                    coords_for_this_infant.append(coords)
                    progress_bar.update(1)
                    curr_iter = curr_iter + 1
            else:
                sample_coords.append(coords_for_this_infant)
                sample = sample + 1
                progress_bar.update(1)
                break
    sample_coords.append(coords_for_this_infant)
    return sample_coords

def exclude_additional_joints(coords, joints_index):
    """
    Excludes the joints necessary for the reduction to the 13 shared joints.

    Parameters:
    coords (list): The 'coordinates' row of the dataset's dataframe.

    joints_index (list): The list of joints' indexes to remove.

    Returns:
    (list): New list containing the reduced set of joints only.
    """
    return [np.delete(coord, joints_index, axis=0) for coord in coords]