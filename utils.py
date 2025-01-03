import os
import json
import pandas as pd

def calculate_average_bbox_area(annotation_dir, class_name: str):
    average_bbox_size = 0
    class_annotation = {}
    count = 0
    for file in os.listdir(annotation_dir):
        if (class_name == os.path.splitext(file)[0]):
            class_annotation = pd.read_json(os.path.join(annotation_dir, file))
            break
    for key in class_annotation:
        bboxes = class_annotation[key].get('bboxes',[])
        
        for box in bboxes:
            count = count + 1
            #Bounding boxes stored in COCO format, therefore index 2 and index 3
            #will contain width and heigh
            average_bbox_size = average_bbox_size + box[2] * box[3]
            
    
    return average_bbox_size, count

def combine_annotations(annotation_dir):
    """
    Combine annotations from all JSON files in the annotation directory.
    Returns:
        class_annotations (list): List of DataFrames, one for each class.
        label_map (dict): Maps class names to integer labels.
    """
    class_annotations = []  # Store annotations for each class
    label_map = {}  # Map class names to integer labels
    i = 0

    # Iterate through all JSON files in the annotation directory
    for file in os.listdir(annotation_dir):
        # Load each JSON file as a DataFrame and store it
        class_annotations.append(pd.read_json(os.path.join(annotation_dir, file)))
        # Map the class name (file name without extension) to an integer
        label_map[os.path.splitext(file)[0]] = i
        i += 1

    return class_annotations, label_map

def count_files_in_directory(dir):
    subdir = os.listdir(dir)
    count = 0

    count = len([file for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))])
    
    print(f"Count per class: {count} for {dir}")            
    return count

def get_class_names(annotation_dir):
    class_names = []

    for file in os.listdir(annotation_dir):
        class_names.append(os.path.splitext(file)[0])
    return class_names

def get_class_paths(dataset_dir):
    class_paths = []

    for subdir in os.listdir(dataset_dir):
        class_paths.append(dataset_dir+"\\"+subdir)
    return class_paths

def get_label_map(annotation_dir):
    label_map = {}  # Map class names to integer labels
    i = 0

    # Iterate through all JSON files in the annotation directory
    for file in os.listdir(annotation_dir):
        
        # Map the class name (file name without extension) to an integer
        label_map[os.path.splitext(file)[0]] = i
        i += 1

    return label_map

def split_path(img_path):
    # Extract the class name (directory) from the image path
    split_path = os.path.split(img_path)
    split_path = os.path.split(split_path[0])
    class_name = split_path[1]
    
    # Get the image key (filename without extension)
    split_path = os.path.split(img_path)
    image_key = os.path.splitext(split_path[1])[0]
    return image_key, class_name
        

def load_config_file_paths():
    # Get the directory where this script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_directory, 'config.json')
    
    # Open the config file using the absolute path
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    # Retrieve paths from the config file
    dataset_path = config.get("dataset_path")
    ann_path = config.get("annotation_path")
    return dataset_path, ann_path
