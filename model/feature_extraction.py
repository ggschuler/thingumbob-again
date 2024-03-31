import numpy as np

class Extract:

    def __init__(self, data, window_size, stride_size, num_bins):
        self.data = data
        self.window_size = window_size
        self.stride_size = stride_size
        self.num_bins = num_bins
        
        self.features = self.data.copy()

    def extract_for_each(self, do_histograms):
        """
        Loops over datasets and populates a new dictionary with the extracted features.
        """
        for dataset_name, dataset in self.data.items():
            extracted_features = self.get_features(dataset, do_histograms)

            #if do_histograms:
            #    extracted_features = self.histogram_encode(extracted_features)
#
#
            #self.features[dataset_name] = extracted_features
        
        

    def histogram_encode(self, extracted_features):
        bins = np.linspace(min(ex))
        return 0


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
                feature_extracted_data = histogam_encode()
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

