#Phase 6: Once the best_resnet_pain.pth path or checkpoint.pth has been determined, 
# we can load the best model and evaluate its performance on a test set, by using the saved parameters/wheights 
# from the best model to make predictions on the test set and calculate performance metrics such as MAE and RMSE.,
#best_resnet_pain.pth = your trained AI model


import torch
import torch.nn as nn
import torchvision.models as models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #check if GPU is available, otherwise use CPU

def rebuild_resnet(): 

   
    model = models.resnet18(weights=None) #checkpoint file already contains pre trained knowledge

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2)  # Binary classification: LEFT (0) vs RIGHT (1)
    )

    checkpoint = torch.load("checkpoint.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    #Loading the saved model parameters/wheights from the best model checkpoint,

    epoch = checkpoint["epoch"]
    print(f"Loaded best model checkpoint from epoch {epoch+1}")

    model.eval()

    model.to(device)

    return model


#since we are predicitng a continous pain score, we use regression loss

#Load the saved optimizer state from the best model checkpoint,

# Optional learning rate scheduler, reduces learning rate if validation loss stops improving


