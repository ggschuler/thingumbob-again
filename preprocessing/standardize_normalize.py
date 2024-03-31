from preprocessing.processing_utils import exclude_additional_joints, find_nearest_frames_with_valid_data, fill_interpolation, find_tri_h_and_line
import numpy as np
import os
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter

class Process:
    """
    Passes the data through pre-processing steps, including inter-datasets standardization and inner-data normalization.
    Steps are:
        1. Standardize number of joints to 13 (@func: standardize_joint_number).
        2. Filter movement signal for removing zero'ed and low confidence score values (@func: remove_zeroes, @func: remove_low_CIs).
        3. Interpolate on relevant signal gaps using Piecewise Cubic Hermite Interpolation Polynomial (@func: interpolate_nan_values).
        4. Re-scale data with reference to standard vector length from skeleton features (@func: rescale).
        5. Pivot data with reference to Nose joint (@func: pivot).
        6. Rotate data with reference to standard angle from skeleton features (@func: rotate).
        7. Smooth movement signal using rolling-window median filter (@func: smooth).
        8. Normalize with MinMax method (@func: minmax).
        
    """
    def __init__(self, load_paths, save_paths):
        self.load_paths = load_paths
        self.save_paths = save_paths
        self.joint_names = ['Nose', 'L-Shoulder', 'R-Shoulder', 'L-Elbow', 'R-Elbow', 'L-Wrist', 'R-Wrist', 'L-Hip', 'R-Hip', 'L-Knee', 'R-Knee', 'L-Feet', 'R-Feet']
        self.unwanted_joints_from_pmigma = [1,2,3,4]
        self.unwanted_joints_from_openpose = [1,8,15,16,17,18,19,20,21,22,23,24]
        self.minirgbd_rvi38_connections = [
            (0,1), (1,2), (2,3),
            (0,4), (4,5), (5,6),
            (0,7), (7,8), (8,9),
            (0,10), (10,11), (11,12)
        ]
        self.pmigma_connections = [
                    (0, 1), (1, 3), (3, 5),
                    (0, 2), (2, 4), (4, 6),
                    (0, 7), (0, 8),
                    (7, 9), (9, 11),
                    (8, 10), (10, 12)
        ]

    def process_data(self):
        """
        Performs all pre-processing steps on datasets.
        """
        minirgbd = pd.read_pickle(os.path.join(self.save_paths['MINI-RGBD'], 'MINI-RGBD.pkl'))
        pmigma   = pd.read_pickle(os.path.join(self.save_paths['PMI-GMA']  , 'PMI-GMA.pkl'))
        rvi38    = pd.read_pickle(os.path.join(self.save_paths['RVI-38']   , 'RVI-38.pkl'))
        datasets = [minirgbd, pmigma, rvi38]
        datasets_iterator = iter(self.save_paths.items())
        for raw_dataset in datasets:
            save_where = next(datasets_iterator)
            print(f'Processing {save_where[0]}:')
            if raw_dataset.equals(pmigma):
                ref =  'pmigma_output'
            else:
                ref = 'openpose_output'
            standardized_dataset = self.standardize_joint_number(raw_dataset, ref)
            zerofiltered_dataset = self.remove_zeroes(standardized_dataset, ref)
            CIfiltered_dataset   = self.remove_low_CIs(zerofiltered_dataset, ref)
            interpolated_dataset = self.interpolate_nan_values(CIfiltered_dataset)
            rescaled_dataset     = self.rescale(interpolated_dataset, ref)
            pivoted_dataset      = self.pivot(rescaled_dataset)
            rotated_dataset      = self.rotate(pivoted_dataset, ref)
            smoothed_dataset     = self.smooth(rotated_dataset, 5)
            normalized_dataset   = self.minmax(smoothed_dataset)
            normalized_dataset.to_pickle(os.path.join(save_where[1], save_where[0]+'_processed.pkl'))
            print(f'{save_where[0]} saved in {save_where[1]}.')

    def standardize_joint_number(self, dataset, ref):
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
            mod_dataset['coordinates'] = mod_dataset['coordinates'].apply(lambda c: exclude_additional_joints(c, self.unwanted_joints_from_pmigma))
        elif ref == 'openpose_output':
            mod_dataset['coordinates'] = mod_dataset['coordinates'].apply(lambda c: exclude_additional_joints(c, self.unwanted_joints_from_openpose))
        return mod_dataset

    def remove_zeroes(self, dataset, ref):
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

    def remove_low_CIs(self, dataset, ref):
        """
        This function remove pose data coordinates where the associated confidence score is below a threshold.
        If ref == 'pmigma_output', there is no CIs and the input dataset is returned.

        Parameters:
        dataset (pd.dataframe): The dataframe containing the dataset's data.

        Returns: 
        (pd.dataframe): The modified datataset, containing the remaining pose data for each sample in the dataset.
        """
        alln = 0
        currdatan = 0
        laterdatan = 0
        if ref == 'pmigma_output':
            return dataset
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
        return mod_data

    def interpolate_nan_values(self, dataset):
        """
        Does Piecewise Cubic Hermite Interpolation Polynomial for all datapoints which are NaNs in the dataset.

        Parameters:
        dataset (pd.dataframe): The dataframe containing the dataset's data.

        Returns: 
        (pd.dataframe): The modified datataset, containing all samples's PCHIP-interpolated movement signals.
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
        mod_dataset = fill_interpolation(mod_dataset)    
        return mod_dataset

    def rescale(self, dataset, ref):
        """
        Rescales the pose data to a standard reference scale, i.e. the height of 
        the triangle formed by the nose, left hip, and right hip, in the (arbitrary) 100th frame.

        Parameters:
        dataset (pd.dataframe): The dataframe containing the dataset's data.

        ref (string): Whether the output is from a traditional OpenPose format or unique to PMI-GMA.

        Returns: 
        (pd.dataframe): The modified datataset, containing the rescaled data.
        """ 
        mod_dataset = dataset.copy()
        for index, row in mod_dataset.iterrows():
            coords = np.array(row['coordinates'])
            if ref == 'openpose_output':
                h, _ = find_tri_h_and_line(coords, 0, 7, 10)
            elif ref == 'pmigma_output':
                h, _ = find_tri_h_and_line(coords, 0, 7, 8)
            scaled_coords = coords / h
            mod_dataset.at[index, 'coordinates'] = scaled_coords
        return mod_dataset

    def pivot(self, dataset):
        """
        Pivot the data with reference to the nose joint for every frame. As a result, the
        nose data is 0 for all frames, and other joints' coordinates are relative to it.

        Parameters:
        dataset (pd.dataframe): The dataframe containing the dataset's data.

        Returns: 
        (pd.dataframe): The modified datataset, containing the pivoted data.
        """ 
        mod_dataset = dataset.copy()
        for index, row in mod_dataset.iterrows():
            coords = np.array(row['coordinates'])
            pivot_coords = coords[:, 0]
            relative_coords = coords - pivot_coords[:, None]
            mod_dataset.at[index, 'coordinates'] = relative_coords
        return mod_dataset

    def rotate(self, dataset, ref):
        """
        Rotates the pose data to a standard reference scale, i.e. the angle between the 
        vector formed by the nose, left hip, and right hip triangle, in the (arbitrary) 
        100th frame, and the x-axis.

        Parameters:
        dataset (pd.dataframe): The dataframe containing the dataset's data.

        ref (string): Whether the output is from a traditional OpenPose format or unique to PMI-GMA.

        Returns: 
        (pd.dataframe): The modified datataset, containing the rotated data.
        """ 
        mod_dataset = dataset.copy()
        for index, row in mod_dataset.iterrows():
            coords = np.array(row['coordinates'])
            if ref == 'pmigma_output':
                _, line = find_tri_h_and_line(coords, 0, 7, 8)
            elif ref == 'openpose_output':
                _, line = find_tri_h_and_line(coords, 0, 7, 10)
            theta = np.arctan2(line[1], line[0])
            trans_coords = coords
            rotation_matrix = np.array([[np.cos(-theta), -np.sin(-theta)],
                                        [np.sin(-theta), np.cos(-theta)]])
            rotated_coords = np.dot(trans_coords[:,:,:2], rotation_matrix.T)
            mod_dataset.at[index,'coordinates'] = rotated_coords
        return mod_dataset

    def smooth(self, dataset, window_size):
        """
        Passes the pose data through a rolling-window median filter of specific window size.

        Parameters:
        dataset (pd.dataframe): The dataframe containing the dataset's data.

        window_size (int): Size of the rolling-window.

        Returns: 
        (pd.dataframe): The modified datataset, containing the smoothed data.
        """ 
        mod_dataset = dataset.copy()
        smoothed_coords = []
        for _, row in mod_dataset.iterrows():
            coords = np.array(row['coordinates'])
            filt_coords = np.zeros_like(coords)
            for joint in range(coords.shape[1]):
                for coord in range(coords.shape[2]):
                    filt_coords[:, joint, coord] = median_filter(coords[:, joint, coord], size=window_size, mode='mirror')
            smoothed_coords.append(filt_coords)
        mod_dataset['coordinates'] = smoothed_coords
        return mod_dataset

    def minmax(self, dataset):
        """
        Perform MinMax normalization for each sample.

        Parameters:
        dataset (pd.dataframe): The dataframe containing the dataset's data.

        Returns: 
        (pd.dataframe): The modified datataset, containing the normalized (final-stage) data.
        """
        mod_dataset = dataset.copy()
        for index, row in mod_dataset.iterrows():
            coords = row['coordinates']
            min_val = np.min(coords)
            max_val = np.max(coords)
            minmaxed = (coords - min_val) / (max_val - min_val)
            mod_dataset.at[index, 'coordinates'] = minmaxed
        return mod_dataset