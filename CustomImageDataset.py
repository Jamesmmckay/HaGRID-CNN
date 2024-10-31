import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
import json
import os
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

class CustomImageDataset(Dataset):
    def __init__(self, annotation_dir, img_dir, transform=None, target_transform=None):
        """
        Initialize the dataset with annotations, image directory, and optional transformations.
        """
        self.annotations = []
        self.label_map = {}
        # Load and combine annotations from all JSON files in the annotation directory
        self.annotations, self.label_map = combine_annotations(annotation_dir)
        self.transform = transform

        # Gather all image file paths from the directory structure
        self.image_paths = []
        class_names = os.listdir(img_dir)
        
        for class_name in class_names:
            class_dir = os.path.join(img_dir, class_name)
            for file in os.listdir(class_dir):
                # Store the full path to each image
                self.image_paths.append(os.path.join(class_dir, file))

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get an image and its corresponding annotations, transformations, and target (bounding box + label).
        """
        img_path = self.image_paths[idx]

        # Extract the class name (directory) from the image path
        split_path = os.path.split(img_path)
        split_path = os.path.split(split_path[0])
        class_name = split_path[1]
        
        # Get the image key (filename without extension)
        split_path = os.path.split(img_path)
        image_key = os.path.splitext(split_path[1])[0]

        # Retrieve the annotation for the current image
        image_annotation = self.annotations[self.label_map[class_name]].get(image_key)
        image = Image.open(img_path)

        # Extract bounding box from the annotation
        bbox = image_annotation["bboxes"][0]

        # Retrieve label: the gesture class for the image
        if (image_annotation.iloc[1])[0] != "no_gesture":
            label = self.label_map[(image_annotation.iloc[1])[0]]  # First gesture label
        else:
            label = self.label_map[(image_annotation.iloc[1])[1]]  # Backup label if "no_gesture"

        label = torch.tensor(label)  # Convert label to a tensor for PyTorch operations
        
        try:
            # Apply transformation to the image, if any
            image = self.transform(image)
        except:
            # Log any transformation failures
            print(f"Could not transform the following image: \n{img_path}")
            dir_path = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(dir_path, "log/image_exception_list.txt"), "a") as f:
                f.write(f"{img_path}\n")
            return None  # Skip the faulty image

        # Determine image dimensions, depending on whether it's a PIL image or a tensor
        if isinstance(image, Image.Image):
            image_width, image_height = image.size  # For PIL images
        else:
            image_width, image_height = image.shape[2], image.shape[1]  # For tensors (C, H, W in PyTorch)

        # Convert bounding box coordinates from relative (0-1) to absolute pixel values
        bbox_pixel = torch.tensor([
            bbox[0] * image_width,
            bbox[1] * image_height,
            bbox[2] * image_width,
            bbox[3] * image_height
        ])

        # Validate bounding box dimensions (width, height should be positive)
        x_min, y_min, width, height = bbox_pixel
        if width <= 0 or height <= 0:
            print(f"Invalid bounding box: {bbox_pixel.numpy()} (Image size: {image_width}x{image_height})")
            return None  # Skip this sample or handle the invalid bounding box case

        # Prepare the target output: bounding box and label
        target = {
            "boxes": bbox_pixel.unsqueeze(0),  # Convert bbox to shape (1, 4) for a single box
            "label": label
        }

        return image, label

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
