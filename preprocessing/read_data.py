import os
import pandas as pd
from scipy.io import loadmat
from preprocessing.processing_utils import process_pmi_samples, read_from_openpose

class Read:
    """
    Reads data from file. Works with unaltered MINI-RGBD, PMI-GMA, and RVI-38 folders as obtained with authors in May 2023.
    """
    def __init__(self, load_paths, save_paths):
        self.load_paths = load_paths
        self.save_paths = save_paths

    def read_data(self):
        print(f"Loading MINI-RGBD data")
        self.load_minirgbd()
        print(f"Loading PMI-GMA data")
        self.load_pmigma()
        print(f'Loading RVI-38 data')
        self.load_rvi38()

    def load_minirgbd(self):
        labels = loadmat(os.path.join(self.load_paths['MINI-RGBD'], 'labels.mat'))
        labels = labels['labels'][0]
        true_file_list = os.listdir(os.path.join(self.load_paths['MINI-RGBD'], 'MRGBD'))
        cut_file_list = [i.split('_00000')[0] for i in true_file_list]
        samples = [i[16:] for i in sorted(set(cut_file_list))]
        sample_coords = read_from_openpose(true_file_list, 'MINI-RGB_Export_{}', samples, os.path.join(self.load_paths['MINI-RGBD'], 'MRGBD'))
        sample_ids = ['MINI_RGBD{}'.format(i) for i in samples]
        data = pd.DataFrame(index=sample_ids, data={'label':labels, 'coordinates':sample_coords})
        data.index.name = 'ID'
        data.to_pickle(os.path.join(self.save_paths['MINI-RGBD'], 'MINI-RGBD.pkl'))

    def load_pmigma(self):
        labels = pd.read_csv(os.path.join(self.load_paths['PMI-GMA'],'joint_points.txt'), sep=' ', names=['child_id', 'label'], index_col='child_id')
        labels = labels.sort_index()
        labels['ID'] = labels.index.to_series().apply(lambda x: x.split('_')[0]).map(int)
        labels = labels.drop_duplicates(subset=['ID'], keep='last')
        labels['full_child_ID'] = labels.index
        labels.index = labels['ID']
        labels = labels.drop(['ID'], axis=1)
        keypoints_411 = os.path.join(self.load_paths['PMI-GMA'], '411')
        keypoints_709 = os.path.join(self.load_paths['PMI-GMA'], '709')
        coords_411 = process_pmi_samples(keypoints_411)
        coords_709 = process_pmi_samples(keypoints_709)
        coords = coords_411 + coords_709
        pre_data = pd.DataFrame(coords, columns=['full_child_id', 'coordinates'])
        pre_data['ID'] = pre_data['full_child_id'].str.split('_').str[0].astype(int)
        pre_data['label'] = pre_data['ID'].map(labels['label'])
        pre_data = pre_data.set_index('full_child_id')
        pre_data = pre_data.sort_index()
        data = pre_data[['label', 'coordinates', 'ID']]
        data = data.rename(columns={'ID':'simpl_ID'})
        data.index.name = 'ID'
        data.index = ['PMIGMA_{}'.format(i) for i in data.index]
        data.index.name = 'ID'
        data.to_pickle(os.path.join(self.save_paths['PMI-GMA'], 'PMI-GMA.pkl'))

    def load_rvi38(self):
        labels = loadmat(os.path.join(self.load_paths['RVI-38'], 'RVI_38_labels.mat'))
        labels = labels['labels'][0]
        true_file_list = os.listdir(os.path.join(self.load_paths['RVI-38'], '25J_RVI38_Full_Processed'))
        cut_file_list  = [i.split('_00000')[0] for i in true_file_list]
        samples = [i.split('RVI_38_')[1] for i in sorted(set(cut_file_list ))]
        sample_coords = read_from_openpose(true_file_list, 'RVI_38_{}', samples, os.path.join(self.load_paths['RVI-38'], '25J_RVI38_Full_Processed'))
        data = pd.DataFrame(index=range(1, 39),  data={'label':labels, 'coordinates':sample_coords})
        data.index = ['RVI38_{:02d}'.format(i) for i in data.index]
        data.index.name = 'ID'
        #save to pkl.
        data.to_pickle(os.path.join(self.save_paths['RVI-38'], 'RVI-38.pkl'))