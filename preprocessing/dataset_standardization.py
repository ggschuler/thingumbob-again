import pandas as pd
from read_data import save_paths
from scipy.interpolate import PchipInterpolator
from utils import exclude_additional_joints, find_nearest_frames_with_valid_data
import os
import numpy as np

joint_names = ['Nose', 'L-Shoulder', 'R-Shoulder', 'L-Elbow', 'R-Elbow', 'L-Wrist', 'R-Wrist', 'L-Hip', 'R-Hip', 'L-Knee', 'R-Knee', 'L-Feet', 'R-Feet']
minirgbd = pd.read_pickle(os.path.normpath(save_paths['MINI-RGBD']))
pmigma   = pd.read_pickle(os.path.normpath(save_paths['PMI-GMA']  ))
rvi38    = pd.read_pickle(os.path.normpath(save_paths['RVI-38']   ))
unwanted_joints_from_openpose = [1,8,15,16,17,18,19,20,21,22,23,24]

def main():
    datasets = [minirgbd, pmigma, rvi38]
    for dataset in datasets:
        if dataset.equals(pmigma):
            ref =  'pmigma_output'
        else:
            ref = 'openpose_output'
        a = standardize_joint_number(dataset, ref)
        b = remove_zeroes(a, ref)
        c = remove_low_CIs(b)
        d = interpolate_nan_values(c)
        #check_interpolate(dataset)
        #rescale(dataset)
        #pivot(dataset)
        #rotate(dataset)
        #smooth(dataset)
        #minmax(dataset)

def standardize_joint_number(dataset, ref):
    """
    This function reduces the number of joints per sample to 13, for every dataset. 
    See @joint_names for the list of remaining joints.

    Parameters:
    dataset (pd.dataframe): The dataframe containing the dataset's data.

    ref (string): Whether the output is from a traditional OpenPose format or unique to PMI-GMA.
    
    Returns: 
    (pd.dataframe): The modified datataset, containing the reduced joints for each sample in the dataset.
    """
    mod_dataset = dataset.copy()
    if ref == 'pmigma_output':
        mod_dataset['coordinates'] = mod_dataset['coordinates'].apply(lambda c: exclude_additional_joints(c, [1,2,3,4]))
    elif ref == 'openpose_output':
        mod_dataset['coordinates'] = mod_dataset['coordinates'].apply(lambda c: exclude_additional_joints(c, unwanted_joints_from_openpose))
    return mod_dataset

def remove_zeroes(dataset, ref):
    """
    Removes all zeroes from the pose data, which usually corresponds to error-like output.

    Parameters:
    dataset (pd.dataframe): The dataframe containing the dataset's data.

    ref (string): Whether the output is from a traditional OpenPose format or unique to PMI-GMA.

    Returns:
    (pd.dataframe): The modified datataset, containing the w/o-zero data for each sample in the dataset.
    """
    mod_dataset = dataset.copy()
    for index, row in mod_dataset.iterrows():
        coords = np.array(row['coordinates'])
        for frame in range(coords.shape[0]):
            for joint in range(coords.shape[1]):
                if ref == 'pmigma_output':
                    current = coords[frame, joint, 0:2]
                    if (current == 0).any():
                        coords[frame, joint] = [np.nan, np.nan]
                elif ref == 'openpose_output':
                    current = coords[frame, joint, 0:2]
                    if (current == 0).any():
                        coords[frame, joint] = [np.nan, np.nan, np.nan]
        mod_dataset.at[index, 'coordinates']  =  coords 
    mod_dataset = mod_dataset.dropna(subset=['coordinates'])
    return mod_dataset

def remove_low_CIs(dataset):
    """
    This function remove pose data coordinates where the associated confidence score is below a threshold.

    Parameters:
    dataset (pd.dataframe): The dataframe containing the dataset's data.
    
    Returns: 
    (pd.dataframe): The modified datataset, containing the remaining pose data for each sample in the dataset.
    """
    alln = 0
    currdatan = 0
    laterdatan = 0

    mod_data = dataset.copy()
    for index, row in mod_data.iterrows():
        coords = np.array(row['coordinates'])
        alln += (coords[:,:,2]).size
        currdatan += np.sum(np.isnan(coords[:,:,2]))
        for joint in range(coords.shape[1]):
            mean_ci = np.nanmean(coords[:,joint,2])
            for frame in range(coords.shape[0]):
                if coords[frame, joint, 2] < (mean_ci * 0.7):
                    coords[frame, joint] = [np.nan, np.nan, np.nan]
                else:
                    coords[frame, joint] = coords[frame, joint]
            mod_data.at[index, 'coordinates'] = coords
            laterdatan += np.sum(np.isnan(coords[:,:,2]))
    
    
    mod_data = mod_data.dropna(subset=['coordinates'])

def interpolate_nan_values(dataset):
    """
    This function remove pose data coordinates where the associated confidence score is below a threshold.

    Parameters:
    dataset (pd.dataframe): The dataframe containing the dataset's data.
    
    Returns: 
    (pd.dataframe): The modified datataset, containing the remaining pose data for each sample in the dataset.
    """
    mod_dataset = dataset.copy()
    for index,row  in mod_dataset.iterrows():
        coords = np.array(row['coordinates'])
        for joint in range(coords.shape[1]):
            for frame in range(coords.shape[0]):
                if np.isnan(coords[frame, joint]).any():
                    left_frame, right_frame = find_nearest_frames_with_valid_data(coords, frame, joint)
                    if left_frame is not None and right_frame is not None:
                        x_values = np.concatenate([coords[left_frame, joint, 0], coords[right_frame, joint, 0]])
                        y_values = np.concatenate([coords[left_frame, joint, 1], coords[right_frame, joint, 1]])
                        frame_numbers = np.concatenate([left_frame, right_frame])
                        sorted_indices = np.argsort(frame_numbers)
                        frame_numbers_sorted = frame_numbers[sorted_indices]
                        x_values_sorted = x_values[sorted_indices]
                        y_values_sorted = y_values[sorted_indices]
                        x_interp = PchipInterpolator(frame_numbers_sorted, x_values_sorted)
                        y_interp = PchipInterpolator(frame_numbers_sorted, y_values_sorted)
                        coords[frame, joint, 0] = x_interp(frame)
                        coords[frame, joint, 1] = y_interp(frame)
        mod_dataset.at[index, 'coordinates']  =  coords        
    return mod_dataset







if __name__=='__main__':
    main()