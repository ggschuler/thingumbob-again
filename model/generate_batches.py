import numpy as np
from sklearn.model_selection import train_test_split
import torch
from enum import Enum

class Setup(Enum):
    SINGLEMINI = 1
    SINGLERVI  = 2
    SINGLEPMI  = 3
    MINIRVIPMI = 4

def get_setup_type(setups):
    if all(dataset in setups for dataset in ['MINI-RGBD', 'RVI-38', 'PMI-GMA']):
        return Setup.MINIRVIPMI
    elif 'MINI-RGBD' in setups:
        return Setup.SINGLEMINI
    elif 'RVI-38' in setups:
        return Setup.SINGLERVI
    elif 'PMI-GMA' in setups:
        return Setup.SINGLEPMI


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, features, sample_rate):
        self.samples = list()
        self.train = list()
        self.val = list()
        self.test = list()

        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.features = features
        self.sample_rate = sample_rate
        self.data = 0

    def reset(self):
        self.index = 0
        self.samples = self.samples.sample(frac=1).reset_index(drop=True)

    def has_next(self):
        if self.index < len(self.train):
            return True
        return False

    def read_data(self, left_out, data, test=False):
        def create_inner_split(total_samples, positive_samples):
            positive_data = []
            negative_data = []
            for i in range(len(self.train)):
                label = self.train.iloc[i, 1]
                print(label)
                if label.max() == 1:
                    positive_data.append(self.train.iloc[i])
                else:
                    negative_data.append(self.train.iloc[i])

            positive_data = np.random.permutation(positive_data)
            negative_data = np.random.permutation(negative_data)
            sel_positive = positive_data[:positive_samples]
            sel_negative = negative_data[:total_samples - positive_samples]
            inner_val = np.concatenate((sel_positive, sel_negative))
            remaining_positive = positive_data[positive_samples:]
            remaining_negative = negative_data[total_samples - positive_samples:]
            inner_train = np.concatenate((remaining_positive, remaining_negative))

            return inner_train, inner_val

        self.samples = self.features
        self.data = data
        # FOR MINI-RGBD
        if data==Setup.SINGLEMINI:
          print('mini')
          if test == False:
            self.test = self.samples.iloc[left_out]
            self.train = self.samples.drop(self.samples.index[left_out])
            print(self.test)
            print(self.train)
            self.train, self.val = create_inner_split(total_samples=2, positive_samples=1)
            df = self.test.to_frame().T
            self.test = df.to_numpy()
            print(self.train.shape, self.test.shape)
          elif test == True:
            self.test = self.samples.iloc[left_out]
            self.train = self.samples.drop(self.samples.index[left_out])
            self.train, self.val = create_inner_split(total_samples=2, positive_samples=1)
            df = self.test.to_frame().T
            self.test = df.to_numpy()
            self.train = np.vstack((self.train, self.val))
            self.val = self.test

        if data==Setup.SINGLERVI:
            print('rvi')
            if test == False:
              train_df, val_df = train_test_split(self.samples, test_size=0.2, stratify=self.samples['labels'], random_state=42)
              val_df, test_df = train_test_split(val_df, test_size=0.2, stratify=val_df['labels'], random_state=42)
              self.test = np.column_stack((np.array(test_df['features']), np.array(test_df['labels'])))
              self.train = np.column_stack((np.array(train_df['features']), np.array(train_df['labels'])))
              self.val = np.column_stack((np.array(val_df['features']), np.array(val_df['labels'])))
            elif test == True:
              train_df, val_df = train_test_split(self.samples, test_size=0.2, stratify=self.samples['labels'], random_state=42)
              val_df, test_df = train_test_split(val_df, test_size=0.2, stratify=val_df['labels'], random_state=42)
              self.test = np.column_stack((np.array(test_df['features']), np.array(test_df['labels'])))
              self.train = np.column_stack((np.array(train_df['features']), np.array(train_df['labels'])))
              self.val = np.column_stack((np.array(val_df['features']), np.array(val_df['labels'])))
              self.train = np.vstack((self.train, self.val))
              self.val = self.test

            print(f'T:{self.train.shape}:{np.sum(self.train[:,1])}, V:{self.val.shape}:{np.sum(self.val[:,1])}, TE:{self.test.shape}:{np.sum(self.test[:,1])}')

        if data==Setup.SINGLEPMI or data==Setup.MINIRVIPMI:
            print('pmi')
            train_df, val_df = train_test_split(self.samples, test_size=0.2, stratify=self.samples['labels'], random_state=42)
            val_df, test_df = train_test_split(val_df, test_size=0.2, stratify=val_df['labels'], random_state=42)
            self.test = np.column_stack((np.array(test_df['features']), np.array(test_df['labels'])))
            self.train = np.column_stack((np.array(train_df['features']), np.array(train_df['labels'])))
            self.val = np.column_stack((np.array(val_df['features']), np.array(val_df['labels'])))
            print(f'T:{self.train.shape}:{np.sum(self.train[:,1])}, V:{self.val.shape}:{np.sum(self.val[:,1])}, TE:{self.test.shape}:{np.sum(self.test[:,1])}')
        else:
            print('no data type?')


    def next_batch(self, batch_size):
        batch = self.train[self.index:self.index + batch_size]

        self.index += batch_size
        batch_input = []
        batch_target = []
        i = 0
        for vid in batch:
            try:
                features = vid[0]
            except IOError:
                print('stop')
            classes = np.zeros(np.shape(features)[1], dtype=int)
            for i in range(len(classes)):
                classes[i] = int(vid[1])
            batch_input.append(features[:, ::self.sample_rate, :])
            batch_target.append(classes[::self.sample_rate])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), features.shape[2], max(length_of_sequences), 13, dtype=torch.float) 
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        sample_weight = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1], :] = torch.from_numpy(batch_input[i].transpose(2,1,0))
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
        return batch_input_tensor, batch_target_tensor, mask, sample_weight

    def next_batch_test(self, batch_size):
        batch = self.val[:]
        self.index += batch_size
        batch_input = []
        batch_target = []
        for vid in batch:
            try:
                features = vid[0]
            except IOError:
                print('stop')
            classes = np.zeros(np.shape(features)[1], dtype=int)
            for i in range(len(classes)):
                classes[i] = int(vid[1])
            batch_input.append(features[:, ::self.sample_rate, :])
            batch_target.append(classes[::self.sample_rate])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), features.shape[2], max(length_of_sequences), 13, dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)

        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        sample_weight = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1], :] = torch.from_numpy(batch_input[i].transpose(2,1,0))
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
        return batch_input_tensor, batch_target_tensor, mask, sample_weight