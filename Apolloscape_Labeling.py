import argparse
import os
import glob
import numpy as np
import pickle


def process_cone_labels(folder_path, pcd_folder_path, output_file='processed_labels.pkl', extra = False, max_dist = 40):
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
    extra_cones = []
    d = 0
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

                content = f.readlines()
                cones = {}
                for i, line in enumerate(content):
                    if line.startswith('5'):
                        # Extract the cone information
                        cone = [float(x) for x in line.split()[1:4]]
                        dist = np.sqrt(cone[0]**2 + cone[1]**2)
                        if dist <= max_dist:
                            if extra:
                                extra_cones.append([float(x) for x in line.split()[4:7]])

                            cones[f"cone_{d}"] = {
                                "class": "Cone",
                                "location": cone
                            }
                        else:
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
    if extra:
        import matplotlib.pyplot as plt
        distances = np.array(extra_cones)

        heights = distances[:, 2]
        diameters = np.sqrt(distances[:, 0] ** 2 + distances[:, 1] ** 2)

        # Define number of bins for better resolution
        num_bins = 50

        # Create histogram bins for height and diameter with limits
        height_bins = np.linspace(0, 1, num_bins)
        diameter_bins = np.linspace(0, 1, num_bins)

        # Calculate the 2D histogram
        histogram, height_edges, diameter_edges = np.histogram2d(
            heights, diameters, bins=[height_bins, diameter_bins])

        # Plot the 2D histogram as a heatmap
        plt.figure(figsize=(10, 7))
        plt.imshow(histogram.T, origin='lower', aspect='auto',
                   extent=[height_edges[0], height_edges[-1], diameter_bins[0], diameter_bins[-1]],
                   cmap='viridis')
        plt.colorbar(label='Count')
        plt.xlabel('Height')
        plt.ylabel('Diameter')
        plt.title('Cone Histogram: Height vs. Diameter')
        plt.xlim(0.2, 0.9)
        plt.ylim(0.1, 0.7)
        plt.show()
    else:
        with open(output_file, 'wb') as f:
            pickle.dump(label_data, f)
    print(f"found {d} cones out of range! out of {len(extra_cones)+d}")
    return label_data


def open_pickle(file_path):
    with open(file_path, 'rb') as f:
        label_data = pickle.load(f)
    return label_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre process Apolloscape data')
    parser.add_argument('--data_root')
    args = parser.parse_args()
    label_folder = os.path.join(args.data_root, 'detection_train_label')
    pcd_folder = os.path.join(args.data_root, 'PCD')
    output_file = os.path.join(args.data_root, 'PCD_MAP.pkl')

    label_data = process_cone_labels(label_folder, pcd_folder, output_file,True)

