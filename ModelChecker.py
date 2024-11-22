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

import json

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

subsample_path = config.get("file_path_1")
ann_subsample_path = config.get("file_path_2")

batch_size = 10 # these values still need to be adjusted
lr = 0.1
num_classes = 18
epochs = 1
testing_ratio = 1
random_seed = 37

#Define the transformations we would like for the images
transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224))])

#Update Path as needed for the dataset
external_training_path = subsample_path
external_annotation_path = ann_subsample_path

#Get the data set
dataset = CustomImageDataset(annotation_dir=external_annotation_path,img_dir=external_training_path, transform=transform)

#Split the data set into training and testing
dataset_size = len(dataset)
indices = list(range(dataset_size))

split = int(np.floor(testing_ratio * dataset_size))
np.random.seed(random_seed)
np.random.shuffle(indices)

training_indices, testing_indices = indices[split:], indices[:split]

training_sampler = SubsetRandomSampler(training_indices)
testing_sampler = SubsetRandomSampler(testing_indices)


training_loader = DataLoader(dataset, batch_size=batch_size, sampler=training_sampler)
testing_loader = DataLoader(dataset, batch_size=batch_size, sampler=testing_sampler)

#Create variables to track things
train_losses = []
test_losses = []
train_correct = []
test_correct = []

model = AlexNet(num_classes=num_classes)
model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Models", weights_only=True)))

#Loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr)


start_time = time.time()

for epoch in range(epochs):
    trn_cor = 0
    epoch_loss = 0
    tst_cor = 0
    loss_plot = list()

    model.eval()
    
    with torch.no_grad():
        for b,(X_test, y_test) in enumerate(testing_loader):
            
            b+=1
            if (X_test) is None:
                continue
            y_val = model(X_test)

            predicted = torch.max(y_val.data,1)[1]
            loss = criterion(y_val, y_test)
            tst_cor += (predicted == y_test).sum()
            if (b % 20 == 0):
                print(f"Batch: {b}")
            
    
    print(f"Epoch: {epoch} \t Loss: {loss.item()}")
    test_losses.append(loss)
    test_correct.append(tst_cor)

current_time = time.time()
total = current_time - start_time


#torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.realpath(__file__)), "Models"))
#print(f"{test_losses}")
#print("-------------------------------------------------")
#print(f"{test_correct}")
print(f"Training took: {total/60} minutes")
#plt.plot(len(dataset), train_correct, label="Training Correct")
#plt.plot(len(dataset), train_losses, label="Training Losses")
#plt.plot(len(dataset), test_correct, label = "Test Correct")
#plt.plot(len(dataset), test_losses, label= "Test Losses")

#plt.legend()
#plt.show()