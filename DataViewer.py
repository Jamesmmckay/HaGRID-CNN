import utils
import matplotlib.pyplot as plt
import os
import numpy as np

dataset_path, ann_path = utils.load_config_file_paths()

class_names = utils.get_class_names(ann_path)
dataset_class_paths = utils.get_class_paths(dataset_path)
class_count = {}
average_bbox_per_class = {}
bounding_box_size = {}



for class_path in dataset_class_paths:
    
    class_count[os.path.split(class_path)[1]] = utils.count_files_in_directory(class_path)
    average_bbox_area, count = utils.calculate_average_bbox_area(ann_path, os.path.split(class_path)[1])
    average_bbox_per_class[os.path.split(class_path)[1]] = count / class_count[os.path.split(class_path)[1]]
    bounding_box_size[os.path.split(class_path)[1]] = (average_bbox_area / count if count > 0 else 0)


classes = list(class_count.keys())
values = list(class_count.values())

fig = plt.figure(figsize = (10,5))

plt.bar(classes, values, color='maroon', width=0.4)

plt.xlabel("Image Classes")
plt.ylabel("Number of image files")

plt.title("The number of images across the different classes")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

fig2 = plt.figure(figsize=(10,5))
classes = list(bounding_box_size.keys())
values = list(bounding_box_size.values())

plt.bar(classes,values,color='maroon', width=0.4)
plt.xlabel("Image Classes")
plt.ylabel("Average Size of Bounding Boxes")

plt.title("The size of the bounding boxes across the different classes")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

fig3 = plt.figure(figsize=(10,5))
classes = list(average_bbox_per_class.keys())
values = list(average_bbox_per_class.values())


plt.bar(classes,values,color='maroon', width=0.4)
plt.xlabel("Image Classes")
plt.ylabel("Average Number of Bounding Boxes per Image")

plt.title("The number of bounding boxes per image across the different classes")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()