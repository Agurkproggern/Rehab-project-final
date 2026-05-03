#Phase 1: Image data processing
#Phase 2: Train / Val data split
#Phase 3: Data augmentation and dataloader setup
#Phase 4: ResNet model setup
#Phase 5: Training loop, validation loop, MAE / RMSE tracking, saving the best model
# 
# First step of the pipeline is to process the image data. 
# ResNET model expects images to be of size 224x224 and normalized
#  in a specific way. This code will handle that preprocessing.

#Article documentation on image database retrived from: https://pmc.ncbi.nlm.nih.gov/articles/PMC10615976. 
#IMAGES DATABASE: https://osf.io/3hgca/overview?view_only=12b04cd8164d4a6784c04b8c83bf95fb

#Images downloaded from: https://osf.io/3hgca/files/acmus?view_only=12b04cd8164d4a6784c04b8c83bf95fb

#PEMF (Pain E-Motion Faces Database)
#272 video clips, 68 subjects
#Includes:
#posed pain
#spontaneous pain (laser / algometer)
#Includes FACS coding + intensity labels

#PROS: 

#   -High-quality annotations
#   -Mix of real + posed

#CONS: 
#   -Smaller dataset
#   -Sometimes requires digging into supplementary materials to download

#Normative ratings of pain intensity, valence and arousal were provided by students of three different
#  European universities. 
# Six independent coders carried out a coding process on the facial stimuli
#  based on the Facial Action Coding System (FACS), in which ratings of 
# intensity of pain, valence and arousal were computed for each type of facial expression.
#The dataset includes PAIN INTENSITY that is scaled (from 0, “no pain”, to 8, “greatest imaginable pain”),
#Each participant was exposed to 4 different conditions: 
# One Neutral expression and three pain-related facial expressions: 1. posed, 2. spontaneous-algometer and 3. spontaneous-CO2 laser


#This project will focus initially on the POSED PAIN expressions,
#  as they are more standardized and easier to work with for initial model training. 
# The spontaneous expressions can be more variable and may require more complex modeling techniques 
# to accurately capture the nuances of pain expression.



import os 
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from torchvision import transforms


def process_images_head_pose(database_path="HeadPoseImageDatabase"):
    """
    Process images from the Head Pose Image Database.
    Extracts head tilt direction (left/right) from filename angles.
    
    Args:
        database_path: Path to the HeadPoseImageDatabase folder
    
    Returns:
        DataFrame with columns: 'image_path', 'tilt_direction'
        where tilt_direction is 0 (LEFT) or 1 (RIGHT)
    """
    image_df = pd.DataFrame(columns=['image_path', 'tilt_direction'])
    
    if not os.path.exists(database_path):
        raise FileNotFoundError(f"Database path not found at {database_path}")
    
    # Iterate through each person folder (Person01, Person02, etc.) directly in database path
    person_dirs = sorted([d for d in os.listdir(database_path) if d.startswith('Person')])
    
    for person_dir in person_dirs:
        person_path = os.path.join(database_path, person_dir)
        
        if not os.path.isdir(person_path):
            continue
        
        # List all image files in the person directory (txt, jpg, pgm formats)
        image_files = sorted([f for f in os.listdir(person_path) 
                             if f.endswith(('.txt', '.jpg', '.pgm'))])
        
        for image_file in image_files:
            # Extract angles from filename
            # Filename format: person##(frame)(vertical)(sign)(horizontal).txt/.jpg/.pgm
            # Example: person01100-90+0.txt means: vertical=-90, horizontal=+0
            match = re.search(r'person\d+\d+([+-]\d+)([+-]\d+)', image_file)
            
            if not match:
                continue
            
            vertical_angle = int(match.group(1))      # e.g., -90 (pitch)
            horizontal_angle = int(match.group(2))    # e.g., +0 (yaw) - This determines LEFT/RIGHT
            
            # Determine tilt direction based on horizontal angle (yaw)
            # Negative angle = LEFT tilt (label 0)
            # Positive angle = RIGHT tilt (label 1)
            tilt_direction = 0 if horizontal_angle < 0 else 1
            
            image_path = os.path.join(person_path, image_file)
            
            new_row = pd.DataFrame({
                'image_path': [image_path],
                'tilt_direction': [tilt_direction],
                'horizontal_angle': [horizontal_angle]
            })
            image_df = pd.concat([image_df, new_row], ignore_index=True)
    
    print(f"Total images loaded: {len(image_df)}")
    print(f"LEFT tilt (0): {(image_df['tilt_direction'] == 0).sum()}")
    print(f"RIGHT tilt (1): {(image_df['tilt_direction'] == 1).sum()}")
    
    return image_df[['image_path', 'tilt_direction']]

def train_val_data_split(image_df, test_size=0.2, random_state=42):
    """
    Split data by PERSON to ensure train and validation sets have different people.
    This prevents data leakage when multiple frames from same person are used.
    
    Args:
        image_df: DataFrame with 'image_path' and 'tilt_direction' columns
        test_size: Proportion of persons to use for validation (default 0.2)
        random_state: Random seed for reproducibility (default 42)
    
    Returns:
        train_df, val_df: DataFrames with completely different persons in each split
    """
    # Create a copy to avoid modifying original
    df = image_df.copy()
    
    # Extract person ID from image path (e.g., "Person01" from: HeadPoseImageDatabase/Front/Person01/...)
    df['person_id'] = df['image_path'].str.extract(r'(Person\d+)')
    
    # Get unique persons
    unique_persons = df['person_id'].unique()
    
    # Split PERSONS (not images) randomly
    train_persons, val_persons = train_test_split(
        unique_persons,
        test_size=test_size,
        random_state=random_state  # Ensures reproducibility
    )
    
    # Split dataframe: all images from a person go to either train OR val
    train_df = df[df['person_id'].isin(train_persons)].drop('person_id', axis=1)
    val_df = df[df['person_id'].isin(val_persons)].drop('person_id', axis=1)
    
    print(f"Train: {len(train_persons)} persons, {len(train_df)} images")
    print(f"Val: {len(val_persons)} persons, {len(val_df)} images")
    
    return train_df, val_df


import torch
from torchvision import transforms

class AddGaussianNoise:
    """
    Custom transform to add Gaussian noise for data augmentation.
    Helps the model become more robust to variations in input data.
    """
    def __init__(self, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    # RandomResizedCrop randomly crops a portion of the input image and resizes it 
    # to a specified size (224x224 pixels). Helps model generalize to different scales.
    transforms.RandomHorizontalFlip(),
    # RandomHorizontalFlip randomly flips images horizontally for augmentation
    transforms.RandomAffine(degrees=8, translate=(0.03, 0.03), scale=(0.95, 1.05)),
    # RandomAffine applies random rotations, translations, and scaling
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    # ColorJitter randomly changes brightness and contrast
    transforms.ToTensor(),
    AddGaussianNoise(std=0.02),
    # Adding Gaussian noise improves model robustness and reduces overfitting
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        # ImageNet normalization constants for ResNet
    )
])

val_transform = transforms.Compose([
    # Validation transforms are kept simple and consistent to provide reliable benchmarks
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    # ResNet models typically expect input images to be of size 224x224 pixels
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class HeadTiltDataset(Dataset):
    """
    Dataset class for head pose images.
    Loads images and returns them with their tilt direction label (0=LEFT, 1=RIGHT).
    """
    def __init__(self, image_df, transform=None):
        self.image_df = image_df
        self.transform = transform

    def __len__(self):
        return len(self.image_df)

    def __getitem__(self, idx):
        row = self.image_df.iloc[idx]
        
        # Try to read as image first (for actual images)
        image_path = row["image_path"]
        
        # Check if it's a .txt file (which references image data)
        if image_path.endswith('.txt'):
            # For the head pose database, we may need to read image data from binary or create synthetic display
            # For now, we'll try to find corresponding image files
            # The database structure may require customization based on actual format
            image_path_jpg = image_path.replace('.txt', '.jpg')
            image_path_pgm = image_path.replace('.txt', '.pgm')
            
            if os.path.exists(image_path_jpg):
                image = cv2.imread(image_path_jpg)
            elif os.path.exists(image_path_pgm):
                image = cv2.imread(image_path_pgm, cv2.IMREAD_GRAYSCALE)
                # Convert grayscale to RGB for ResNet compatibility
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                # If no image file found, create a placeholder or raise error
                raise FileNotFoundError(f"No image file found for {image_path}")
        else:
            image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get tilt direction label (0=LEFT, 1=RIGHT)
        label = row["tilt_direction"]
        
        if self.transform:
            image = self.transform(image)
        
        # Return image and label as long tensors for classification (not regression)
        return image, torch.tensor(label, dtype=torch.long)


from torch.utils.data import DataLoader

def create_dataloaders(image_df, batch_size=32):
    """
    Create training and validation dataloaders from image dataframe.
    
    Args:
        image_df: DataFrame with 'image_path' and 'tilt_direction' columns
        batch_size: Number of images per batch (default 32)
    
    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    
    # Split data by person (no need to balance - head tilts are already balanced)
    train_df, val_df = train_val_data_split(image_df)
    
    # Create datasets with appropriate transforms
    train_dataset = HeadTiltDataset(train_df, transform=train_transform)
    val_dataset = HeadTiltDataset(val_df, transform=val_transform)
    
    # Create dataloaders for batched iteration
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # DataLoader provides efficient batch loading and shuffling for training
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataloaders created:")
    print(f"Train batches: {len(train_loader)} (batch_size={batch_size})")
    print(f"Val batches: {len(val_loader)} (batch_size={batch_size})")
    
    return train_loader, val_loader




