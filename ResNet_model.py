#Phase 4: Setup ResNet Model
#This file sets up the ResNet model for head tilt direction classification
#Binary classification task: LEFT tilt (label 0) vs RIGHT tilt (label 1)
#This document defines the model, loss function, and optimizer for training


import torch
import torch.nn as nn
import torchvision.models as models


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Check if GPU is available, otherwise use CPU

# Load pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#Using pretrained weights from ResNet18 trained on ImageNet
#This allows us to leverage learned features which improves performance especially with smaller datasets

# Replace final classifier with binary classification head
#Output 2 class probabilities for LEFT (0) vs RIGHT (1) tilt
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 2)  # Binary classification: 2 output neurons
)

# Freeze most layers to leverage pretrained features
#Only fine-tune the last layer and the classification head
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4 for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True

# Unfreeze the new classification head
for param in model.fc.parameters():
    param.requires_grad = True

# Move model to device
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
#For binary classification, CrossEntropyLoss expects:
# - Outputs: raw logits with shape (batch_size, 2)
# - Targets: class indices with shape (batch_size,) containing 0 or 1

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
    #Learning rate of 1e-4 is appropriate for fine-tuning pretrained models
)

# Optional learning rate scheduler
#Reduces learning rate if validation loss stops improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

print(model)
print("Using device:", device)

