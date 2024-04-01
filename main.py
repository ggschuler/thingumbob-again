from preprocessing.read_data import Read
from preprocessing.standardize_normalize import Process
from model.data_loading import Load
from model.feature_extraction import Extract
from model.generate_batches import BatchGenerator, Setup, get_setup_type
from model.trainer import Trainer
import torch
import random
import pandas as pd



def main():
    def train_val_run(setup):
        batch_generator = BatchGenerator(num_classes=2, actions_dict={0:0, 1:1}, features=features, sample_rate=1)
        trainer = Trainer(dil=[1,2,4,8,16,32,64,128,256,512], 
                          num_f_maps=hyperparams['ftm'],
                          dim=2, num_classes=2, connections=connections, pool=hyperparams['pool'])

        batch_generator.read_data(left_out=fold, data=setup, test=False)
        performance = trainer.train(batch_gen=batch_generator,
                        num_epochs=50,
                        batch_size=batch_size,
                        learning_rate=hyperparams['lr'],
                        device=device,
                        threshold=hyperparams['thr'],
                        weight_decay=hyperparams['wd'], 
                        num_bin = hyperparams['nbins'], 
                        num_f_maps = hyperparams['ftm'], 
                        pool = hyperparams['pool']
                      )
        
        ALL_OPTIM.append(performance)
        performance = performance['W.F1'][0]
        if performance > best_performance:
            best_performance = performance
            best_hyperparams = hyperparams
        
    def test_run(setup):
        print(f'Best hyperparams: {best_hyperparams}')
        print(f'Best performance: {best_performance}')
        final_df = pd.concat(ALL_OPTIM, ignore_index = True)
        batch_generator = BatchGenerator(num_classes=2, actions_dict={0:0, 1:1}, features=features, sample_rate=1)
        trainer = Trainer(dil=[1,2,4,8,16,32,64,128,256,512], 
                          num_f_maps=best_hyperparams['ftm'],
                          dim=2, num_classes=2, connections=connections, pool=best_hyperparams['pool'])

        batch_generator.read_data(left_out=fold, data=setup, test=False)
        performance = trainer.train(batch_gen=batch_generator,
                        num_epochs=50,
                        batch_size=batch_size,
                        learning_rate=best_hyperparams['lr'],
                        device=device,
                        threshold=best_hyperparams['thr'],
                        weight_decay=best_hyperparams['wd'], 
                        num_bin = best_hyperparams['nbins'], 
                        num_f_maps = best_hyperparams['ftm'], 
                        pool = best_hyperparams['pool']
                      )
        print(performance)
        all_folds.append((final_df, performance))
    # load_paths carry the folder on which each dataset was stored when obtained. 
    # If folder names have changed, alter here
    load_paths = {'MINI-RGBD': r".\data\MINI-RGBD\00_25J_MRGB",
                  'PMI-GMA'  : r".\data\PMI-GMA\pmi-gma\joint_points",
                  'RVI-38'   : r".\data\RVI-38"}

    # saves the read data (.pkl file) into the designated folder
    save_paths = {'MINI-RGBD': r".\data\MINI-RGBD",
                  'PMI-GMA'  : r".\data\PMI-GMA",
                  'RVI-38'   : r".\data\RVI-38"}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sliding_window_size = 30
    sliding_window_stridesize = 30

    hyperparams_rang = {
        'lr': [0.01, 0.001, 0.0001],
        'wd': [1e-2, 1e-3, 1e-4],
        'thr': [0.3, 0.4, 0.5],
        'ftm': [32, 64, 128],
        'nbins': [6, 12, 18],
        'pool': ['max', 'avg']
    }

    num_random_search_trials = 40
    best_performance = float('-inf')
    best_hyperparams = None

    reader = Read(load_paths, save_paths)
    processer = Process(load_paths, save_paths)
    loader = Load(load_paths, save_paths)
    #reader.read_data()
    #processer.process_data()

    data = loader.load_data()
    feature_extractor = Extract(data, window_size=sliding_window_size, stride_size=sliding_window_stridesize, do_histograms=True, num_bins=16)
    setup = ['MINI-RGBD']
    features = feature_extractor.generate_featurespace(setup)
    ALL_OPTIM = []

    s = get_setup_type(setup)
    if s == Setup.SINGLEMINI:
        all_folds = []
        for fold in range(len(features)):
            print(f'FOLD {fold}')
            for trial in range(num_random_search_trials):
                print(f'Trial {trial+1}/{num_random_search_trials}')
                hyperparams = {
                    'lr': random.choice(hyperparams_rang['lr']),
                    'wd': random.choice(hyperparams_rang['wd']),
                    'thr': random.choice(hyperparams_rang['thr']),
                    'ftm': random.choice(hyperparams_rang['ftm']),
                    'nbins': random.choice(hyperparams_rang['nbins']),
                    'pool': random.choice(hyperparams_rang['pool'])
                }

                batch_size  = 1
                connections = processer.minirgbd_rvi38_connections
                train_val_run(s)
            test_run(s)

    if s == Setup.SINGLERVI:
       all_folds = []
       for iteration in range(20):
            print(f'ITERATION {iteration}')
            for trial in range(num_random_search_trials):
                print(f'Trial {trial+1}/{num_random_search_trials}')
                hyperparams = {
                    'lr': random.choice(hyperparams_rang['lr']),
                    'wd': random.choice(hyperparams_rang['wd']),
                    'thr': random.choice(hyperparams_rang['thr']),
                    'ftm': random.choice(hyperparams_rang['ftm']),
                    'nbins': random.choice(hyperparams_rang['nbins']),
                    'pool': random.choice(hyperparams_rang['pool'])
                }

                batch_size  = 1
                connections = processer.minirgbd_rvi38_connections
                train_val_run(s)
            test_run(s)

    if s == Setup.SINGLEPMI:
        all_folds = []
        for iteration in range(5):
            print(f'ITERATION {iteration}')
            for trial in range(num_random_search_trials):
                print(f'Trial {trial+1}/{num_random_search_trials}')
                hyperparams = {
                    'lr': random.choice(hyperparams_rang['lr']),
                    'wd': random.choice(hyperparams_rang['wd']),
                    'thr': random.choice(hyperparams_rang['thr']),
                    'ftm': random.choice(hyperparams_rang['ftm']),
                    'nbins': random.choice(hyperparams_rang['nbins']),
                    'pool': random.choice(hyperparams_rang['pool'])
                }

                batch_size  = 1
                connections = processer.pmigma_connections
                train_val_run(s)
            test_run(s)

if __name__=='__main__':
    main()