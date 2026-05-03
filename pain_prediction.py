#PHASE 7: Evaluation and prediction of pain intensity using the trained ResNet model
#Goal: Input image → preprocessing → ResNet → pain score (0–1 or 0–10)
#With a total of phase 7, we now have a working AI pain system 


import torch
import torch.nn as nn
import torchvision.models as models
from pipeline import Resnet_rebuild as trained_resnet

# Configure matplotlib backend for Windows compatibility
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for Windows GUI support
import matplotlib.pyplot as plt

# Import PIL for image loading (more compatible than OpenCV)
from PIL import Image


# Load model with error handling
try:
    trained_model = trained_resnet.rebuild_resnet()
    device = trained_resnet.device
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    raise

from torchvision import transforms #Defining transforms for input images

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

import cv2
print("✓ Transform and CV2 loaded")


def predict_tilt(image_path, model, transform, device):
    """
    Predict head tilt direction (LEFT or RIGHT) from an image.
    
    Args:
        image_path: Path to input image
        model: Trained ResNet model
        transform: Image preprocessing transforms
        device: torch device (CPU or GPU)
        
    Returns:
        str: "LEFT" or "RIGHT"
    """
    print(f"  [1] Loading image from: {image_path}")
    # Loading image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    print(f"  [2] Image loaded: shape {image.shape}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"  [3] Color converted to RGB")

    # Apply transform
    print(f"  [4] Applying transforms...")
    image = transform(image)
    print(f"  [5] Transform applied, shape: {image.shape}")

    # Add batch dimension
    print(f"  [6] Adding batch dimension...")
    image = image.unsqueeze(0).to(device)
    print(f"  [7] Batch dimension added, shape: {image.shape}")

    # Predict
    print(f"  [8] Running model inference...")
    with torch.no_grad():
        output = model(image)  # Shape: [1, 2] - logits for LEFT and RIGHT classes
        print(f"  [9] Model output shape: {output.shape}")
        print(f"  [10] Model output values: {output}")
        
        predicted_class = torch.argmax(output, dim=1).item()  # Get class index: 0=LEFT, 1=RIGHT
        print(f"  [11] Predicted class index: {predicted_class}")

    # Convert class index to label
    tilt_direction = "LEFT" if predicted_class == 0 else "RIGHT"
    print(f"  [12] Final prediction: {tilt_direction}")

    return tilt_direction


def show_prediction(image_path, head_tilt, display=True):
    """
    Display image with predicted head tilt direction.
    
    Args:
        image_path: Path to input image
        head_tilt: Prediction result ("LEFT" or "RIGHT")
        display: If False, skip display (useful for headless environments)
    """
    if not display:
        print(f"Prediction result: {head_tilt}")
        return
    
    try:
        # Use PIL for image loading (better Windows compatibility)
        image = Image.open(image_path)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(f"Predicted Head Tilt: {head_tilt}", fontsize=14, fontweight='bold')
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"⚠ Could not display image: {e}")
        print(f"Prediction result: {head_tilt}")