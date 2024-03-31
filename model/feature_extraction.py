import numpy as np

class Extract:

    def __init__(self, data, window_size, stride_size):
        self.data = data
        self.window_size = window_size
        self.stride_size = stride_size
        
        self.features = self.data.copy()

    def extract_for_each(self):
        for dataset_name, dataset in self.data.items():
            self.features[dataset_name] = self.get_features(dataset)
        
    def get_features(self, dataset):
        for _, row in dataset.iterrows():
            coords = np.array(row['coordinates'])
            disps = self.windowed_displacement(coords)
            motion_orientation = self.windowed_orientation(coords)
            feature_extracted_data = np.concatenate([disps, motion_orientation], axis=-1)
        return feature_extracted_data

    def windowed_displacement(self, coords):
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

