#Phase 5: Training the ResNet model for head tilt classification
#This file trains the ResNet model on the head tilt direction dataset
#training loop
#validation loop
#accuracy tracking
#saving the best model

#At this point, we have: train_loader, val_loader, setup ResNet model,
# loss criterion (CrossEntropyLoss) and an optimizer

import torch
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pipeline import ResNet_model as resnet
from pipeline import image_data_processing as imp
import matplotlib.pyplot as plt


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        # outputs shape: (batch_size, 2) - logits for 2 classes
        loss = criterion(outputs, labels)

        loss.backward()
        #Backpropagation step - computes gradients of loss wrt model parameters
        optimizer.step()
        #Updates model parameters based on computed gradients and learning rate

        running_loss += loss.item() * images.size(0)
        #Accumulate total loss for the epoch

        # Get predicted class by taking argmax of outputs
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    #Average loss per sample for the epoch
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate accuracy and other classification metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return epoch_loss, accuracy, precision, recall, f1


def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # outputs shape: (batch_size, 2) - logits for 2 classes
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Get predicted class by taking argmax of outputs
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate accuracy and classification metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return epoch_loss, accuracy, precision, recall, f1


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                scheduler=None, num_epochs=30):
    
    best_val_loss = float("inf")

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": []
    }

    for epoch in range(num_epochs):
        train_loss, train_acc, train_prec, train_recall, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_prec, val_recall, val_f1 = validate_one_epoch(
            model, val_loader, criterion, device
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["train_precision"].append(train_prec)
        history["train_recall"].append(train_recall)
        history["train_f1"].append(train_f1)
        
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val   F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            }, "checkpoint.pth")
            print("✓ Saved best model checkpoint.")

        print("-" * 60)

    return history







