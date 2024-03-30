import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

joint_names = ['Nose', 'L-Shoulder', 'R-Shoulder', 'L-Elbow', 'R-Elbow', 'L-Wrist', 'R-Wrist', 'L-Hip', 'R-Hip', 'L-Knee', 'R-Knee', 'L-Feet', 'R-Feet']

def visualize_mean_confidence_across_joints(dataset):
    """
    This plots the mean confidence score (as outputted from OpenPose) for each joint
    across all sequence, for each sample. Standardly, this may only be called with
    the MINI-RGBD or RVI-38 datasets, and not with PMI-GMA.

    Parameters:
    dataset (pd.dataframe): The dataframe containing the dataset's data.

    Returns: 
    None
    """
    all_confs = []
    path_p1 = dataset.index[0].split('_')[0]
    
    for sample in range(len(dataset)):
        conf_ints_for_this_sample = []
        n_frames = range(len(dataset['coordinates'][sample]))
        for joint in range(len(dataset['coordinates'][sample][0])):
            conf_int_for_joint = [[x for x in dataset['coordinates'][sample][frame][joint]] for frame in n_frames]
            conf_int_for_joint = [x[2] for x in conf_int_for_joint] 
            mean_conf_for_joint = np.mean(conf_int_for_joint)
            conf_ints_for_this_sample.append(mean_conf_for_joint)
        all_confs.append(conf_ints_for_this_sample)
        
    all_confs = np.array(all_confs)
    all_confs_T = np.transpose(all_confs)
    conf_per_joints_allsamples = np.mean(all_confs_T, axis=1)
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.heatmap(all_confs_T, square=True, annot=False, linewidth=.5, fmt=".2f", xticklabels=False, yticklabels=False)
    ax.set_xticks(np.arange(len(dataset)) + 0.5)
    labels = dataset['label']
    xtick_labels = [f"{i} ({j})" for i, j in zip(range(1, len(dataset) + 1), labels)]
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=18) 
    ax.set_yticks(np.arange(13) + 0.5)
    ax.set_yticklabels(joint_names, rotation=0, fontsize=18) 
    ax.set_xlabel("Sample nº (label)", labelpad=10)
    ax.set_ylabel("Joint name")
    plt.tight_layout()
    plt.show()

def visualize_coord_signal(dataset, sample, plot_type, which=0):
    """
    This plots the pose data x- and y- components across the sequence of a specific @sample 
    in the @dataset. Can be called to plot all 13 joints signals, or a specific joint.

    Parameters:
    dataset (pd.dataframe): The dataframe containing the dataset's data.

    sample (int): The index of the sample for which the singnal will be plotted.

    plot_type (int): Whether to plot all (=1) or a specific joint (=0).

    which (int): If @plot_type is set to 1, then specifies the joint index.

    Returns: 
    None
    """
    fig, axs = plt.subplots(2,7, figsize=(15,4), sharex=True, sharey=True)
    n_frames = range(len(dataset['coordinates'][sample]))
    if plot_type == 1:
        for index, ax in enumerate(axs.flat):
            if index<13:
                xs = []
                ys = []
                for frame in n_frames:
                    xs.append(dataset['coordinates'][sample][frame][index][0])
                    ys.append(dataset['coordinates'][sample][frame][index][1])
                ax.plot(n_frames, xs, label='X', color="#d53e4f")
                ax.plot(n_frames, ys, label='Y', color="#3288bd")
                ax.set_title('{}'.format(joint_names[index]))
        path_p1 = dataset.index[sample].split('_')[0]
        path_p2 = dataset.index[sample].replace(dataset.index[sample].split('_')[0]+'_', '')
        fig.text(0.5, 0, 'Frame sequence', ha='center')
        fig.text(0.08, 0.5, 'Coordinate data', va='center', rotation='vertical')
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='right', bbox_to_anchor=(0.95, 0.85))
    else:
        fig, ax = plt.subplots()
        xs = []
        ys = []
        for frame in n_frames:
            xs.append(dataset['coordinates'][sample][frame][which][0])
            ys.append(dataset['coordinates'][sample][frame][which][1])
        ax.plot(n_frames, xs, label='X', color="#d53e4f")
        ax.plot(n_frames, ys, label='Y', color="#3288bd")
        path_p1 = dataset.index[sample].split('_')[0]
        path_p2 = dataset.index[sample].replace(dataset.index[sample].split('_')[0]+'_', '')
    plt.show()
    plt.close()

def visualize_skel_frame(dataset, sample, connections):
    """
    This plots the first frame of a sequence of skeleton data, where points
    denote joints and edges denote bones.

    Parameters:
    dataset (pd.dataframe): The dataframe containing the dataset's data.

    sample (int): The index of the sample for which the singnal will be plotted.

    connections (list): A list of tuples whose elements are the index of two adjacent joints. Should contain 13 tuples.
    
    Returns: 
    None
    """
    sns.set_style("white")
    fig, ax = plt.subplots()
    ax.set_title('Sample: {0}'.format(dataset.index[sample]))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    xs = [a[0] for a in dataset['coordinates'][sample][0]]
    ys = [a[1] for a in dataset['coordinates'][sample][0]]
    scatter = ax.scatter(xs, ys, s=20)
    lines = [ax.plot([], [], linestyle='-', linewidth=1)[0] for _ in range(len(connections))]

    for i, (start, end) in enumerate(connections):
        x_start = xs[start]
        y_start = ys[start]
        x_end = xs[end]
        y_end = ys[end]
        lines[i].set_data([x_start, x_end], [y_start, y_end])

    plt.show()