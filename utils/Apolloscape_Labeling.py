import glob
import os
import statistics
import numpy as np

def scan_folder(folder_name, make_folder_flag=False):
  """Scans a folder with the given name, processes subfolders and their txt files.

  Args:
    folder_name: The name of the folder to scan.
  """
  cone_lengths = []
  # Check if the folder exists
  if not os.path.exists(folder_name):
    print(f"Error: Folder '{folder_name}' not found.")
    return

  # Get a list of subfolders
  subfolders = glob.glob(os.path.join(folder_name, '*'))

  # Process each subfolder
  for vid in subfolders:
    print(f"Processing subfolder: {vid}")
    # Keep the subfolder name in memory (you can use a list or dictionary)
    # For example:
    vid_name = os.path.basename(vid)

    out_frames = {}
    # Find txt files in the subfolder
    frames = glob.glob(os.path.join(vid, '*.txt'))

    # Process txt files (e.g., read content, perform operations)
    for frame in frames:
      temp = []
      with open(frame, 'r') as f:
        content = f.readlines()
        for line in content:
          if line[0] == "5":
            # what we keep from the original cone detection
            cone = line.split(" ")[1:4]
            cone_lengths.append(cone_length(cone))
            temp.append(" ".join(cone))

      if len(temp) > 0:
        out_frames[frame.split("\\")[-1]] = "\n ".join(str(elem) for elem in temp)
    if out_frames:
      path = "result\\" + frame.split("\\")[-2]+"\\"
      if make_folder_flag:
        os.makedirs(os.path.dirname(path))
        for f in out_frames:
          with open(path + f, 'w') as file:
            file.write(out_frames[f])
  return cone_lengths
def cone_length(arr):
  len = float(arr[0])**2 + float(arr[1])**2 + float(arr[2])**2
  return len ** 0.5

def distance_histogram(distances, bin_width=5):
  """Creates a histogram of distances with given bin width.

  Args:
    distances: A list or numpy array of float distances.
    bin_width: The width of each distance bin.

  Returns:
    A list of tuples (bin_start, bin_end, count).
  """

  # Convert to numpy array for efficiency
  distances = np.array(distances)

  # Calculate bin edges
  bin_edges = np.arange(0, distances.max() + bin_width, bin_width)

  # Create histogram
  hist, _ = np.histogram(distances, bins=bin_edges)

  # Format output
  histogram_data = {}
  for i in range(len(hist)):
    bin_start = bin_edges[i]
    bin_end = bin_edges[i+1]
    count = hist[i]
    histogram_data[int(bin_start)] = count

  return histogram_data

import matplotlib.pyplot as plt

def plot_dict(data):
  """Plots a graph from a dictionary where keys are x-values and values are y-values.

  Args:
    data: A dictionary containing x-y pairs.
  """

  x_values = list(data.keys())
  y_values = list(data.values())

  plt.plot(x_values, y_values)
  plt.xlabel('Distance')
  plt.ylabel('# of cones')
  plt.title('Cones per Distance')
  plt.show()


if __name__ == '__main__':
  x = scan_folder("detection_train_label",False)
  histogram = distance_histogram(x,1)
  plot_dict(histogram)
