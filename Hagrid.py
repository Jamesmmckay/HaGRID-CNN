import numpy as np
import matplotlib.pyplot as plt
import time
import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from AlexNet import AlexNet
from CustomImageDataset import CustomImageDataset
import utils

# Hyperparameters and configuration
batch_size = 35  # Number of images in each batch for training/testing
lr = 0.01  # Learning rate
num_classes = 18  # Number of gesture classes to predict
epochs = 5  # Number of times the model sees the entire dataset
testing_ratio = 0.2  # Percentage of the dataset used for testing
random_seed = 37  # Random seed to ensure reproducibility

# Define transformations for the images (Resize and convert to tensor)
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])

# Update the path for your dataset
external_training_path, external_annotation_path = utils.load_config_file_paths()

# Load the custom dataset with annotations and image transformations
dataset = CustomImageDataset(annotation_dir=external_annotation_path, img_dir=external_training_path, transform=transform)

# Split the dataset into training and testing sets based on the testing ratio
dataset_size = len(dataset)
indices = list(range(dataset_size))

# Compute the split point for training/testing based on the testing ratio
split = int(np.floor(testing_ratio * dataset_size))
np.random.seed(random_seed)  # Ensure random shuffling is consistent
np.random.shuffle(indices)  # Shuffle indices to ensure randomness

# Get indices for training and testing sets
training_indices, testing_indices = indices[split:], indices[:split]

# Create samplers to generate training and testing batches
training_sampler = SubsetRandomSampler(training_indices)
testing_sampler = SubsetRandomSampler(testing_indices)

def collate_fn(batch):
    # Filter out None items
    batch = [item for item in batch if item is not None]
    return torch.utils.data.default_collate(batch)

# Use this collate function in the DataLoader
training_loader = DataLoader(dataset, batch_size=batch_size, sampler=training_sampler, collate_fn=collate_fn)
testing_loader = DataLoader(dataset, batch_size=batch_size, sampler=testing_sampler, collate_fn=collate_fn)

# Initialize lists to track training/testing performance over time
train_losses = []
test_losses = []
train_correct = []
test_correct = []

# Instantiate the AlexNet model with the correct number of classes
model = AlexNet(num_classes=num_classes)

# Set the loss function (CrossEntropy for classification tasks)
criterion = nn.CrossEntropyLoss()

# Set the optimizer (Adam optimizer to adjust model parameters)
optimizer = torch.optim.Adam(model.parameters(), lr)

# Record the start time to measure training duration
start_time = time.time()
print(f"Dataset size: {len(dataset)}")
print(f"DataLoader batches: {len(training_loader)}")

# Training loop over the number of epochs
for epoch in range(epochs):
    trn_cor = 0  # Track correct predictions during training
    epoch_loss = 0  # Track loss during the epoch
    tst_cor = 0  # Track correct predictions during testing
    loss_plot = list()  # Track loss values for plotting

    # Set the model to training mode
    model.train()

    # Training step: Iterate over batches of training data
    for b, (X_train, y_train) in enumerate(training_loader):
        b += 1  # Batch number

        # Skip any None values due to transformation issues
        if X_train is None:
            continue
        
        # Forward pass: Make predictions using the model
        y_pred = model(X_train)
        
        # Compute the loss (difference between predicted and actual labels)
        loss = criterion(y_pred, y_train)

        # Get the predicted class for each image
        predicted = torch.max(y_pred.data, 1)[1]
        
        # Count correct predictions in this batch
        batch_cor = (predicted == y_train).sum()
        trn_cor += batch_cor  # Update correct predictions

        # Backpropagation: Compute gradients and update model weights
        optimizer.zero_grad()  # Zero the gradients to avoid accumulation
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        loss_plot.append(loss.item())  # Track loss for plotting

        # Print batch loss for every 40th batch
        if b % 40 == 0:
            print(f"Epoch: {epoch} \t Batch: {b} \t Loss: {loss.item()}")

    # After epoch, record total loss and correct predictions for training
    train_losses.append(sum(loss_plot) / len(loss_plot))
    train_correct.append(trn_cor.item())

    # Testing step: Set model to evaluation mode (no gradient calculations)
    model.eval()

    with torch.no_grad():  # Disable gradient tracking for evaluation
        test_loss_epoch = 0
        for b, (X_test, y_test) in enumerate(testing_loader):
            b += 1  # Batch number
            
            if X_test is None:
                continue
            
            # Forward pass: Get predictions for the testing data
            y_val = model(X_test)
            
            # Get the predicted class for each test image
            predicted = torch.max(y_val.data, 1)[1]
            
            # Count correct predictions
            tst_cor += (predicted == y_test).sum()
            loss = criterion(y_val, y_test)
            test_loss_epoch += loss.item()

    # Calculate and log the loss for the test set
    
    print(f"Epoch: {epoch} \t Loss: {loss.item()}")
    test_losses.append(test_loss_epoch / len(testing_loader))
    test_correct.append(tst_cor.item())

# Calculate total time taken for training
current_time = time.time()
total = current_time - start_time

# Save the trained model's state to a file
torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.realpath(__file__)), "Models"))

# Print the total training time in minutes
print(f"Training took: {total / 60} minutes")

# Plotting results
epochs_range = range(epochs)

plt.figure(figsize=(10, 5))

# Plot losses
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Losses over Epochs")
plt.legend()

# Plot accuracies
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_correct, label="Train Correct")
plt.plot(epochs_range, test_correct, label="Test Correct")
plt.xlabel("Epochs")
plt.ylabel("Correct Predictions")
plt.title("Correct Predictions over Epochs")
plt.legend()

# Save the plot to a file
output_dir = "Plots"
os.makedirs(output_dir, exist_ok=True)
plot_path = os.path.join(output_dir, "training_results.png")
plt.savefig(plot_path)

plt.show()

print(f"Plot saved to: {plot_path}")