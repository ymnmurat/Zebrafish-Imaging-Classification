import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc


# Here we want to analyze the number of bright green pixels in images from two folders
# We only want the model to be trained on images that have a significant amount of bright green pixels
# Thus we calculate the ratio of bright green pixels in each image, and then only train on images that have a ratio above a certain threshold

def bright_green_ratio(image_path, brightness_threshold):
    """
    Optimized function to calculate the ratio of bright green pixels in an image.
    Uses NumPy for vectorized operations instead of Python loops.
    """
    # Open image and convert to NumPy array
    image = Image.open(image_path)
    img_array = np.array(image)
    
    # Extract RGB channels
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]
    
    # Calculate brightness
    brightness = (r + g + b) / 3
    
    # Create mask for bright green pixels
    # Green > Red AND Green > Blue AND Brightness > threshold
    bright_green_mask = (g > r) & (g > b) & (brightness > brightness_threshold)
    
    # Calculate ratio
    total_pixels = img_array.shape[0] * img_array.shape[1]
    bright_green_count = np.sum(bright_green_mask)
    
    return bright_green_count / total_pixels if total_pixels > 0 else 0

def analyze_folder(folder_path):
    """
    Optimized function to analyze all images in a folder for bright green pixel ratios.
    Includes progress tracking and better error handling.
    """
    image_green_ratio_2 = {}
    
    # Get all .tif files
    tif_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
    total_files = len(tif_files)
    
    print(f"Analyzing {total_files} images in {folder_path}...")
    
    for i, filename in enumerate(tif_files):
        try:
            image_path = os.path.join(folder_path, filename)
            green_ratio = bright_green_ratio(image_path, 68)
            image_green_ratio_2[filename] = green_ratio
            
            # Print progress every 10 files
            if (i + 1) % 10 == 0 or (i + 1) == total_files:
                print(f"Processed {i + 1}/{total_files} images")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            image_green_ratio_2[filename] = 0  # Default to 0 if error occurs
    
    print(f"Completed analysis of {len(image_green_ratio_2)} images")
    return image_green_ratio_2


folder1_ratios = analyze_folder('images/class_0')
folder2_ratios = analyze_folder('images/class_1')

bright_green_ratio_threshold = 0.0000001  # Adjust this threshold as needed
# For testing on small datasets we set it to 0.0, normally we would set it to 0.000001
# # We chose this threshold to remove 30% of the total images

# Define the transformation to apply to each image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Reduced from 750x750 to 224x224 for memory efficiency
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, folder1, folder2, transform=None):
        self.transform = transform
        self.img_labels = []

        # Check if directories exist
        if not os.path.isdir(folder1):
            raise ValueError(f"Directory {folder1} does not exist")
        if not os.path.isdir(folder2):
            raise ValueError(f"Directory {folder2} does not exist")

        # Supported image file extensions
        supported_extensions = ('tif','png', 'jpg', 'jpeg')

        # Label images in folder1 as 0
        for file in os.listdir(folder1):
            if file.lower().endswith(supported_extensions):
                if(folder1_ratios[file] > bright_green_ratio_threshold):
                    # This is to ensure we only include images with significant bright green pixels
                    self.img_labels.append((os.path.join(folder1, file), 0))

        # Label images in folder2 as 1
        for file in os.listdir(folder2):
            if file.lower().endswith(supported_extensions):
                if(folder2_ratios[file] > bright_green_ratio_threshold):
                    # This is to ensure we only include images with significant bright green pixels
                    self.img_labels.append((os.path.join(folder2, file), 1))

        if len(self.img_labels) == 0:
            raise ValueError("No images found in the provided directories")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# Directories containing the images
folder1 = 'images/class_0'  # Change this to the path where your images for label 0 are stored
folder2 = 'images/class_1'  # Change this to the path where your images for label 1 are stored

print("Loading images from folders...")
# Create dataset and dataloader
dataset = CustomImageDataset(folder1=folder1, folder2=folder2, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)  # Reduced batch size and workers

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])

print(f"Training on {len(train_data)} images, testing on {len(test_data)} images")
train_loader = DataLoader(train_data, batch_size=16, shuffle=True) # Reduced batch size for memory efficiency
test_loader = DataLoader(test_data, batch_size=16, shuffle=False) # Reduced batch size for memory efficiency

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        # With 224x224 input and 4 pooling layers: 224 -> 112 -> 56 -> 28 -> 14
        self.fc1 = nn.Linear(64 * 14 * 14, 256)  # Much smaller fully connected layer
        self.fc2 = nn.Linear(256, 2)  # Binary classifier

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(x)  # Additional pooling layer to reduce spatial dimensions
        x = x.view(-1, 64 * 14 * 14)  # Flatten the tensor
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

print("Initializing model...")

# Initialize the model, loss function, optimizer, and learning rate scheduler
model = SimpleCNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 10  # Adjust as needed
train_losses = []
val_losses = []

print("Starting training...")
# Memory optimization: gradient accumulation steps
accumulation_steps = 2  # Effective batch size = 16 * 2 = 32

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()  # Zero gradients at start of epoch
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps
        
        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item() * accumulation_steps:.4f}')
        
        # Memory cleanup
        del images, labels, outputs, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Handle remaining gradients if any
    if len(train_loader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validate the model
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Memory cleanup
            del images, labels, outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    val_loss /= len(test_loader)
    val_losses.append(val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]}')

# Initialize lists to store metrics
accuracies = []
sensitivities = []
specificities = []
balanced_accuracies = []
roc_aucs = []

# Function to calculate metrics
def calculate_metrics(labels, preds):
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    balanced_accuracy = (sensitivity + specificity) / 2
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    return accuracy, sensitivity, specificity, balanced_accuracy, roc_auc

print("Calculating metrics for the final epoch...")
# After training, calculate metrics for the final epoch
model.eval()
all_labels = []
all_preds = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)

        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Calculate metrics
accuracy, sensitivity, specificity, balanced_accuracy, roc_auc = calculate_metrics(all_labels, all_preds)

print(f'Accuracy: {accuracy:.4f}')
print(f'Sensitivity: {sensitivity:.4f}')
print(f'Specificity: {specificity:.4f}')
print(f'Balanced Accuracy: {balanced_accuracy:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')
