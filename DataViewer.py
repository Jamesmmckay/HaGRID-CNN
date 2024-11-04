import utils
import matplotlib.pyplot as plt
import os
import numpy as np

dataset_path, ann_path = utils.load_config_file_paths()

class_names = utils.get_class_names(ann_path)
dataset_class_paths = utils.get_class_paths(dataset_path)
class_count = {}

for class_path in dataset_class_paths:
    class_count[os.path.split(class_path)[1]] = utils.count_files_in_directory(class_path)

classes = list(class_count.keys())
values = list(class_count.values())

fig = plt.figure(figsize = (20,5))

plt.bar(classes, values, color='maroon', width=0.4)

plt.xlabel("Image Classes")
plt.ylabel("Number of image files")

plt.title("The number of images across the different classes")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
