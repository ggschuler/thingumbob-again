from preprocessing.read_data import Read
from preprocessing.standardize_normalize import Process

def main():
    # load_paths carry the folder on which each dataset was stored when obtained. 
    # If folder names have changed, alter here
    load_paths = {'MINI-RGBD': r".\data\MINI-RGBD\00_25J_MRGB",
                  'PMI-GMA'  : r".\data\PMI-GMA\pmi-gma\joint_points",
                  'RVI-38'   : r".\data\RVI-38"}

    # saves the read data (.pkl file) into the designated folder
    save_paths = {'MINI-RGBD': r".\data\MINI-RGBD",
                  'PMI-GMA'  : r".\data\PMI-GMA",
                  'RVI-38'   : r".\data\RVI-38"}
    
    reader = Read(load_paths, save_paths)
    processer = Process(load_paths, save_paths)

    reader.read_data()
    processer.process_data()

if __name__=='__main__':
    main()