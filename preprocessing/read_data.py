import numpy as np
import pandas as pd
import os
from scipy.io import loadmat
import json

# load_paths carry the folder on which each dataset was stored when obtained. 
# If folder names have changed, alter here
load_paths = {'MINI-RGBD': r".\data\MINI-RGBD\00_25J_MRGB",
              'PMI-GMA'  : r".\data\PMI-GMA\pmi-gma\joint_points",
              'RVI-38'   : r".\data\RVI-38\25J_RVI38_Full_Processed"}

# saves the read data (.pkl file) into the designated folder
save_paths = {'MINI-RGBD': r"thingumbob again!\data\MINI-RGBD\MINI-RGBD.pkl",
              'PMI-GMA'  : r"thingumbob again!\data\PMI-GMA\PMI-GMA.pkl",
              'RVI-38'   : r"thingumbob again!\data\RVI-38\RVI-38.pkl"}

def main():
    print(f"Loading MINI-RGBD data")
    load_minirgbd()

def load_minirgbd():

    labels = loadmat(os.path.join(load_paths['MINI-RGBD'], 'labels.mat'))
    labels = labels['labels'][0]
    true_file_list = os.listdir(os.path.join(load_paths['MINI-RGBD'], 'MRGBD'))
    cut_file_list = [i.split('_00000')[0] for i in true_file_list]
    samples = [i[16:] for i in sorted(set(cut_file_list))]
    sample_coords = read_from_openpose(true_file_list, 'MINI-RGB_Export_{}', samples, os.path.join(load_paths['MINI-RGBD'], 'MRGBD'))
    sample_ids = ['MINI_RGBD{}'.format(i) for i in samples]
    data = pd.DataFrame(index=sample_ids, data={'label':labels, 'coordinates':sample_coords})
    data.index.name = 'ID'
    print(data)
    data.to_pickle(save_paths['MINI-RGBD'])


def read_from_openpose(file_list, custom_prefix, samples, keypoints_path):
    """
    This function should read all files from OpenPose, given that they are on a single folder.

    Parameters:
    file_list (list): List containing all name files outputted from OpenPose.
    custom_prefix (string): The string prefixed to the OpenPose standard "Export_()_()_keypoints.json"
    samples (list): List containing the desired samples IDs/names.
    keypoints_path (string): The path to the folder containing the keypoints files.

    Returns:

    """
    print(samples)
    sample_coords = []
    start = 1
    sample = 0
    while(start < len(file_list)):
        coords_for_this_infant = []
        for i in file_list[start-1:]:
            if i.startswith(custom_prefix.format(samples[sample])):
                with open(os.path.join(keypoints_path, i), "r") as file:
                    data = json.load(file)
                    coords = np.array((data['people'][0]['pose_keypoints_2d'] ))
                    coords = coords.reshape(25,3)
                    coords_for_this_infant.append(coords)
                    start = start + 1
            else:
                sample_coords.append(coords_for_this_infant)
                sample = sample + 1
                break
    sample_coords.append(coords_for_this_infant)
    print(len(sample_coords))
    return sample_coords


if __name__=='__main__':
    main()