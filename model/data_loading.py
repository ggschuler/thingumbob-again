import numpy as np
import math
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore", message="The frame.append method is deprecated and will be removed from pandas in a future version.", category=FutureWarning)

class Load:
    """
    Responsible for loading processed .pkl files into coordinate and label arrays for subsequent feature extraction.
    """
    def __init__(self, load_paths, save_paths):
        self.load_paths = load_paths
        self.save_paths = save_paths

        self.data = {}

    def load_data(self):
        """
        Loads the datasets as coordinate and label lists. If more datasets would be added, declare their 
        path and include it in the @datasets variable in the body of the function. Change @load_paths and
        @save_paths accordingly, and name the value with the dataset intended key (string).

        Returns:
        (dictionary): Name - Dataframe pair of each extracted dataset 
        """
        minirgbd = pd.read_pickle(os.path.join(self.save_paths['MINI-RGBD'], 'MINI-RGBD_processed.pkl'))
        pmigma   = pd.read_pickle(os.path.join(self.save_paths['PMI-GMA']  , 'PMI-GMA_processed.pkl'))
        rvi38    = pd.read_pickle(os.path.join(self.save_paths['RVI-38']   , 'RVI-38_processed.pkl'))
        datasets = [minirgbd, pmigma, rvi38]
        datasets_iterator = iter(self.save_paths.items())
        for processed_dataset in datasets:
            coords = []
            labels = []
            if processed_dataset.equals(rvi38):
                processed_dataset = self.augment_rvi(processed_dataset)
            for _, row in processed_dataset.iterrows():
                coords.append(row['coordinates'])
                labels.append(np.repeat(row['label'], row['coordinates'].shape[0]))
            self.data[next(datasets_iterator)[0]] = pd.DataFrame({processed_dataset.columns[1]:coords, processed_dataset.columns[0]:labels},index=processed_dataset.index)
        return self.data

    def augment_rvi(self, rvi_data):
        """
        Performs data augmentation on the RVI-38 dataset, increasing samples from 38 to 124.

        Parameters:
        rvi_data (pd.dataframe): The dataframe containing the relevant dataset.

        Returns:
        (pd.dataframe): The augmented dataset, corresponding to a 2-column, 124 rows Dataframe.
        """
        desired_length = 999
        segments_rvi_data = pd.DataFrame(columns=['label','coordinates'])
        for i in range(len(rvi_data)):
            feature = rvi_data.iloc[i]['coordinates']
            label = rvi_data.iloc[i]['label']
            num_segments = math.ceil(feature.shape[0] / desired_length)
            for segment in range(num_segments-1):
                start_idx = segment * desired_length
                end_idx = (segment + 1) * desired_length
                segment_features = feature[start_idx:end_idx, :, :]
                segments_rvi_data = segments_rvi_data.append({'coordinates': segment_features, 'label': label}, ignore_index=True)
        return segments_rvi_data
