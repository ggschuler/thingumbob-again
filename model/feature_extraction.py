import numpy as np
import pandas as pd

class Extract:
    """
    Performs all tasks related to feature extraction, from displacement and 
    orientation scalars to histogram-encoding and graph representation
    """
    def __init__(self, data, window_size, stride_size, do_histograms, num_bins):
        self.data = data
        self.window_size = window_size
        self.stride_size = stride_size
        self.do_histograms = do_histograms
        self.num_bins = num_bins
        self.features = self.data.copy()

    def generate_featurespace(self, setup):
        """
        Generates the feature space for the selected setup. Standardly, the available setups are:
            - MINI-RGBD
            - RVI-38
            - PMI-GMA
            - MINI+RVI+PMI

        Parameters:
        setup (list): List of strings containing the names of the datasets, as named in @save_paths.

        Returns:
        (pd.dataframe): a Dataframe with the resulting features and labels of the setup. For example, by inputing 
        ['MINI-RGBD', 'RVI-38', 'PMI-GMA'] the resulting dataframe would contain 12+124+1120=1256 samples
        """
        all_features = []
        for dataset in setup:
            features = self.extract_for_each()[dataset]
            features_df = pd.DataFrame({'features':features['coordinates'], 'labels':features['label'].apply(lambda x: x[:self.num_bins])})
            features_df['features'] = features_df['features'].apply(lambda x: np.array(x).transpose(1, 0, 2))
            all_features.append(features_df)
        
        return pd.DataFrame(np.concatenate([i for i in all_features], axis=0), columns=['features', 'labels'])

    def extract_for_each(self):
        """
        Loops over datasets and populates a new dictionary with the extracted features.

        Returns:
        (dictionary): A dictionary whose keys are the datasets' names and values are their corresponding Dataframes.
        """
        for dataset_name, dataset in self.data.items():
            extracted_features = self.get_features(dataset, self.do_histograms)
            self.features[dataset_name] = extracted_features
        return self.features

    def histogramize(self, one_joint_feature):
        """
        Core function to perform histogram-encoding of a vector containing the features for n-windows in one joint.

        Parameters:
        one_joint_feature (np.array): Array of shape (windows) containing the feature-values along n-windows.

        Returns:
        (np.array): Array of shape (@self.num_bin) containing the normalized bin values for the corresponding feature.
        """
        bins = np.linspace(min(one_joint_feature), max(one_joint_feature), self.num_bins+1)
        histogram, _ = np.histogram(one_joint_feature, bins=bins)
        histogram_normalized = histogram / (np.sum(histogram))
        return histogram_normalized
    
    def histogram_along_joint(self, feature_matrix):
        """
        Auxiliary function to perform histogram-encoding joint-by-joint.

        Parameters:
        feature_matrix (np.array): Array of shape (windows, joints) containing the feature matrix of one feature.

        Returns:
        (np.array): Array of shape (joints, @self.num_bin).
        """
        num_frames, _ = feature_matrix.shape
        reshaped_matrix = feature_matrix.reshape(num_frames, -1).T
        histogram_encoded_features = []
        for joint in reshaped_matrix:
            histogram_encoded_feature = self.histogramize(joint)
            histogram_encoded_features.append(histogram_encoded_feature)
        histogram_encoded_matrix = np.array(histogram_encoded_features).T
        return histogram_encoded_matrix

    def histogram_encode(self, extracted_features):
        """
        Auxiliary function to obtain the histogram-encoded features summarized from the window-lenghted extracted features.
        Does this first feature-by-feature and then joint-by-joint.

        Parameters:
        extracted_features (np.array): Array of shape (windows, joints, features) containing the extracted features.

        Returns:
        (np.array): Array of shape (joints, @self.num_bins, 2), where 2 = number of histogram-encoded features.
        """
        _, _, num_features = extracted_features.shape
        histogram_encoded_features = []
        for feature_index in range(num_features):
            feature_matrix = extracted_features[:,:, feature_index]
            histogram_encoded_feature = self.histogram_along_joint(feature_matrix)
            histogram_encoded_features.append(histogram_encoded_feature)
        histogram_encoded_matrix = np.stack(histogram_encoded_features, axis=0)
        histogram_encoded_matrix = histogram_encoded_matrix.T
        return histogram_encoded_matrix

    def get_features(self, dataset, do_histograms):
        """
        Loops over one @dataset's rows and extract features.

        Parameters:
        dataset (pd.dataframe): Dataframe containing one dataset's data (coordinates and labels).

        Returns:
        (pd.dataframe): New Dataframe for the specified dataset's processed data.
        """
        extracted_features = dataset.copy()
        for index, row in extracted_features.iterrows():
            coords = np.array(row['coordinates'])
            disps = self.windowed_displacement(coords)
            motion_orientation = self.windowed_orientation(coords)
            feature_extracted_data = np.concatenate([disps, motion_orientation], axis=-1)
            if do_histograms:
                feature_extracted_data = self.histogram_encode(feature_extracted_data)
            extracted_features.at[index, 'coordinates'] = feature_extracted_data
        return extracted_features

    def windowed_displacement(self, coords):
        """
        Extracts a displacement magnitude between a starting point and an ending point (defined by
        @self.window_size) for every window.

        Parameters:
        coords (np.array): Array of shape (frames, joints, coordinates) containing movement data.

        Returns:
        (np.array): Array of shape (frames, joints, 1), where 1 = displacement scalar.
        """
        T, V, M = coords.shape
        final_sample = np.zeros(((T - self.window_size) // self.stride_size + 1, V, 1))
        for start in range(0, T - self.window_size + 1, self.stride_size):
            end = start + self.window_size
            window_sample = coords[start:end, :, :]
            first_frame_coords = window_sample[0, :, :]
            last_frame_coords = window_sample[-1, :, :]
            delta_x = last_frame_coords[:, 0] - first_frame_coords[:, 0]
            delta_y = last_frame_coords[:, 1] - first_frame_coords[:, 1]
            displacement_distance = np.sqrt(delta_x**2 + delta_y**2) * 100
            displacement_distance = np.expand_dims(displacement_distance, 0)
            displacement_distance = displacement_distance.T
            final_sample[start // self.stride_size, :, :] = displacement_distance
        return final_sample

    def windowed_orientation(self, coords):
        """
        Extracts an orientation angle the vector formed by a starting point and an 
        ending point (defined by @self.window_size) and the x-axis for every window.

        Parameters:
        coords (np.array): Array of shape (frames, joints, coordinates) containing movement data.

        Returns:
        (np.array): Array of shape (frames, joints, 1), where 1 = angle scalar.
        """
        T, V, M = coords.shape
        final_sample = np.zeros(((T - self.window_size) // self.stride_size + 1, V, 1))
        for start in range(0, T - self.window_size + 1, self.stride_size):
          end = start+self.window_size
          window_sample = coords[start:end, :, :]
          first_frame_coords = window_sample[0, :, :]
          last_frame_coords = window_sample[-1, :, :]
          delta_x = last_frame_coords[:, 0] - first_frame_coords[:, 0]
          delta_y = last_frame_coords[:, 1] - first_frame_coords[:, 1]
          motion_angle = np.arctan2(delta_y, delta_x)
          motion_angle = np.degrees(motion_angle)
          motion_angle = np.expand_dims(motion_angle, 0)
          motion_angle = motion_angle.T
          final_sample[start // self.stride_size, :, :] = motion_angle
        return final_sample