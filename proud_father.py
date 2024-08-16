import pickle
import os
import numpy as np
from utils import vis_pc

def load_pickle_file(directory):
    """
    Load a pickle file from the specified directory.

    Parameters:
    directory (str): The directory where the pickle file is located.
    filename (str): The name of the pickle file to load.

    Returns:
    tuple: The contents loaded from the pickle file.
    """
    # Construct the full path to the pickle file
    file_path = os.path.join(directory, 'variables.pkl')

    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return None

    # Load the pickle file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data


# Example usage
if __name__ == "__main__":
    # Specify the directory and file name
    directory = './test_logs/'
    os.makedirs(directory, exist_ok=True)


    # Load the data from the pickle file
    scores_above_70, f1_distribution, time_distribution = load_pickle_file(directory)
    f1_distribution = np.array(f1_distribution)
    time_distribution = np.array(time_distribution)
    key, pc, lidar_bboxes, real_bbox, prediction, recall, f1 = [], [], [], [], [], [], []
    for i, score in enumerate(scores_above_70):
        # [key[i], pc[i], lidar_bboxes[i], real_bbox[i], prediction[i], recall[i], f1[i]] = score
        vis_pc(score[1], score[2], score[3])

    if loaded_data is not None:
        # Unpack the loaded data into separate variables
        loaded_array, loaded_list1, loaded_list2 = loaded_data
        print("Loaded Array:", loaded_array)
        print("Loaded List 1:", loaded_list1)
        print("Loaded List 2:", loaded_list2)
