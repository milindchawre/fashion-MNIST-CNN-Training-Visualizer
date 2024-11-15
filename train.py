import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import json
import os
import numpy as np
from tqdm import tqdm
import logging
from PIL import Image
import io
import base64

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

fashion_mnist_labels = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}

class ConfigurableCNN(nn.Module):
    def __init__(self, channels, kernel_size, activation):
        super(ConfigurableCNN, self).__init__()
        
        # Select activation function
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        else:  # Sigmoid
            self.activation = nn.Sigmoid()
            
        # Create convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=kernel_size, padding=kernel_size//2),
            self.activation,
            nn.MaxPool2d(2),
            nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size, padding=kernel_size//2),
            self.activation,
            nn.MaxPool2d(2),
            nn.Conv2d(channels[1], channels[2], kernel_size=kernel_size, padding=kernel_size//2),
            self.activation,
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[2], 128),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def save_progress(model_id, progress):
    with open(f'static/{model_id}_progress.json', 'w') as f:
        json.dump(progress, f)

def train_model(model_id, config):
    # Extract configuration
    channels = config['channels']
    kernel_size = config['kernel_size']
    activation = config['activation']
    optimizer_name = config['optimizer']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    num_epochs = config['epochs']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training {model_id} on {device}")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    model = ConfigurableCNN(channels, kernel_size, activation).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer based on configuration with learning rate
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:  # SGD
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    training_losses = []
    validation_losses = []
    training_accuracies = []
    test_accuracies = []
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct_train = 0
        total_train = 0
        
        # Training phase
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pred = output.argmax(dim=1)
            correct_train += pred.eq(target).sum().item()
            total_train += target.size(0)
            
            # Update progress bar
            train_acc = 100. * correct_train / total_train
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_acc:.2f}%'
            })
        
        # Calculate training metrics
        avg_train_loss = epoch_loss / len(train_loader)
        train_accuracy = 100. * correct_train / total_train
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        logger.info("Running validation...")
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Validation'):
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(test_loader)
        test_accuracy = 100. * correct / total
        
        # Append metrics
        training_losses.append(avg_train_loss)
        validation_losses.append(avg_val_loss)
        training_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        # Save progress
        progress = {
            'training_losses': training_losses,
            'validation_losses': validation_losses,
            'training_accuracy': training_accuracies,
            'test_accuracy': test_accuracies,
            'current_epoch': epoch,
            'current_batch': -1,  # Reset batch index at epoch end
            'status': 'Training'
        }
        save_progress(model_id, progress)
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        logger.info(f'Test Loss: {avg_val_loss:.4f}, Accuracy: {test_accuracy:.2f}%')
    
    # Generate predictions
    logger.info("Generating sample predictions...")
    model.eval()
    test_samples = []
    with torch.no_grad():
        data_iter = iter(test_loader)
        data, target = next(data_iter)
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        pred = output.argmax(dim=1)
        
        for i in range(10):
            # Convert tensor to PIL Image
            img_array = data[i].squeeze().cpu().numpy()
            # Scale to 0-255 range and invert colors for better visibility
            img_array = ((1 - img_array) * 255).astype(np.uint8)
            # Resize image to be larger (224x224)
            img = Image.fromarray(img_array).resize((224, 224), Image.Resampling.LANCZOS)
            
            # Convert PIL image to base64 string
            buffered = io.BytesIO()
            img.save(buffered, format="PNG", quality=95)  # Increased quality
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            test_samples.append({
                'image': img_str,  # Save base64 string instead of numpy array
                'true': target[i].item(),
                'true_label': fashion_mnist_labels[target[i].item()],
                'pred': pred[i].item(),
                'pred_label': fashion_mnist_labels[pred[i].item()]
            })
    
    # Save results
    np.save(f'static/{model_id}_samples.npy', test_samples)
    torch.save(model.state_dict(), f'static/{model_id}_model.pth')
    
    # Final progress save
    progress['status'] = 'Completed'
    save_progress(model_id, progress)
    
    logger.info("Training completed successfully!")