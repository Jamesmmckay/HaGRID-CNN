import matplotlib.pyplot as plt
import os
from collections import defaultdict
from CustomImageDataset import CustomImageDataset
import torchvision.transforms as transforms

import json

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

subsample_path = config.get("file_path_1")
ann_subsample_path = config.get("file_path_2")
transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224))])


def plot_samples_per_category(dataset):
    """
    Plots the number of samples per category in the given dataset.
    """
    # Dictionary to count images per category
    category_count = defaultdict(int)
    
    # Iterate through all images in the dataset and count them by category
    for img_path in dataset.image_paths:
        class_name = os.path.split(os.path.dirname(img_path))[-1]
        category_count[class_name] += 1

    # Prepare data for plotting
    categories = list(category_count.keys())
    counts = [category_count[category] for category in categories]

    # Plot the bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(categories, counts, color='skyblue')
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.title("Number of Samples per Category", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()  # Adjust layout for better fit
    plt.show()

# Example usage
if __name__ == "__main__":
    external_training_path = subsample_path
    external_annotation_path = ann_subsample_path

    #Get the data set
    dataset = CustomImageDataset(annotation_dir=external_annotation_path,img_dir=external_training_path, transform=transform)

    # Plot the samples per category
    plot_samples_per_category(dataset)
