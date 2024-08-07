import os
import glob
import numpy as np
import pickle


def process_cone_labels(folder_path, pcd_folder_path, output_file='processed_labels.pkl'):
    """
    Processes the cone labels in the given folder, creates a structured dictionary of labels,
    and saves the results to a .pkl file.

    Args:
        folder_path (str): The path to the folder containing the cone label files.
        pcd_folder_path (str): The path to the folder containing the PCD files.
        output_file (str): The path to the output .pkl file.

     the idea of this tool is to take appoloscape folder, and generate data as .pkl file.
     the data is dictionary of dictionaries, this way:
     key: vid#_frame#
     value: dictionary2:
             path: pcd location
             cones: dictionary:
                 ConeID: Dictionary
                     class: Cone
                     location: x,y,z

    """
    label_data = {}
    alone = 0

    # Process each subfolder (vid)
    for vid_folder in glob.glob(os.path.join(folder_path, '*')):
        vid_name = os.path.basename(vid_folder)
        print(f"Processing subfolder: {vid_name}")

        # Process text files in the subfolder
        for frame in glob.glob(os.path.join(vid_folder, '*.txt')):
            frame_name = os.path.splitext(os.path.basename(frame))[0]
            key = f"{vid_name}_{frame_name}"

            # Find corresponding PCD file
            pcd_file = os.path.join(pcd_folder_path, f"result_{vid_name}_frame", f"{frame_name}.pcd")

            if not os.path.exists(pcd_file):
                os.remove(frame)
                continue

            with open(frame, 'r') as f:
                d = 0
                content = f.readlines()
                cones = {}
                for i, line in enumerate(content):
                    if line.startswith('5'):
                        # Extract the cone information
                        cone = [float(x) for x in line.split()[1:4]]
                        cones[f"cone_{d}"] = {
                            "class": "Cone",
                            "location": cone
                        }
                        d = d + 1
            if cones:
                label_data[key] = {
                    "path": pcd_file,
                    "cones": cones
                }
            else:
                os.remove(frame)
                os.remove(pcd_file)
                continue


    # Save the processed labels to a .pkl file
    with open(output_file, 'wb') as f:
        pickle.dump(label_data, f)

    return label_data


def open_pickle(file_path):
    with open(file_path, 'rb') as f:
        label_data = pickle.load(f)
    return label_data


if __name__ == '__main__':
    label_folder = "C:\\Yuval\\Me\\Projects\\Final Project\\Data\\detection_train_label\\detection_train_label"
    pcd_folder = "C:\\Yuval\\Me\\Projects\\Final Project\\Data\\PCD"
    output_file = 'C:\\Yuval\\Me\\Projects\\Final Project\\Data\\PCD_MAP.pkl'

    # label_data = process_cone_labels(label_folder, pcd_folder, output_file)
    # Optionally, you can print some information about the processed data
    # print(f"Processed {len(label_data)} frames")
    test = open_pickle(output_file)
    print(test)




