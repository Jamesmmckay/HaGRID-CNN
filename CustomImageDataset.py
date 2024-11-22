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
import utils

class CustomImageDataset(Dataset):
    def __init__(self, annotation_dir, img_dir, transform=None, target_transform=None):
        """
        Initialize the dataset with annotations, image directory, and optional transformations.
        """
        self.annotations = []
        self.label_map = {}
        self.placeholder_images = {}
        # Load and combine annotations from all JSON files in the annotation directory
        self.annotations, self.label_map = utils.combine_annotations(annotation_dir)
        self.transform = transform

        # Gather all image file paths from the directory structure
        self.image_paths = []
        self.valid_annotations = []
        class_names = os.listdir(img_dir)
         # Log invalid images during initialization
        log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "log/image_exception_list.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        for class_name in class_names:
            class_dir = os.path.join(img_dir, class_name)
            placeholder_set = False  # Track if a placeholder has been set for the class
        
            for file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, file)
                image_key, ___ = utils.split_path(img_path)
                try:
                    # Open and transform the image (only if placeholder isn't set yet)
                    if not placeholder_set:
                        image = Image.open(img_path)
                        if self.transform:
                            image = self.transform(image)
                        
                        # Get annotation and label
                        image_annotation = self.annotations[self.label_map[class_name]].get(image_key)
                        
                        if (image_annotation.iloc[1])[0] != "no_gesture":
                            label = self.label_map[(image_annotation.iloc[1])[0]]
                        else:
                            label = self.label_map[(image_annotation.iloc[1])[1]]
                            
                        label = torch.tensor(label)  # Convert label to tensor
                        self.placeholder_images[class_name] = [image, label]
                        placeholder_set = True  # Mark placeholder as set
        
                    # Add image path to `self.image_paths` regardless of placeholder
                    self.image_paths.append(img_path)
        
                except Exception as e:
                    # Log the invalid image and skip
                    print(f"Error processing {img_path}: {e}")
                    with open(log_path, "a") as f:
                        f.write(f"{img_path}\n")


                    

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

        image_key, class_name = utils.split_path(img_path)

        # Retrieve the annotation for the current image
        image_annotation = self.annotations[self.label_map[class_name]].get(image_key)
        try:
            image = Image.open(img_path)
        except:
            print(f"Could not open the following image: {img_path}")
            dir_path = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(dir_path, "log/image_exception_list.txt"), "a") as f:
                f.write(f"{img_path}\n")
            return self.placeholder_images[class_name][0], self.placeholder_images[class_name][1]  # Skip the faulty image
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
            return self.placeholder_images[class_name][0], self.placeholder_images[class_name][1]  # Skip the faulty image

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
