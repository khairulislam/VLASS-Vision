import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import CosineAnnealingLR

class CNN(L.LightningModule):
    def __init__(self, num_classes, input_shape):
        super(CNN, self).__init__()
        self.save_hyperparameters()
        # input_shape is expected as (height, width, channels) from Keras, 
        # but PyTorch expects (channels, height, width) for Conv2d.
        
        # input -> B x C x H x W
        self.conv1 = nn.Conv2d(1, 75, kernel_size=(3, 3), stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(75)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0) # Keras padding="same" with stride=2 for MaxPool2D is equivalent to PyTorch padding=0 when kernel_size=stride.
        self.conv2 = nn.Conv2d(75, 50, kernel_size=(3, 3), stride=1, padding='same')
        self.dropout1 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        self.conv3 = nn.Conv2d(50, 25, kernel_size=(3, 3), stride=1, padding='same')
        self.bn3 = nn.BatchNorm2d(25)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # To calculate the input features for the first Dense layer, 
        # we need to perform a dummy forward pass or calculate manually.
        # Let's assume a way to calculate it or pass a dummy input.
        # A common practice is to pass a dummy tensor to determine the flattened size.
        # We'll put a placeholder here and explain how to calculate it.
        # For now, let's assume `flattened_size` will be calculated dynamically.
        # self.fc1 = nn.Linear(flattened_size, 512) 

        self.fc1 = nn.Linear(self._get_flattened_size(input_shape), 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

    def _get_flattened_size(self, input_shape):
        # Create a dummy input tensor to calculate the output shape after convolutions and pooling
        # PyTorch expects (batch_size, channels, height, width)
        dummy_input = torch.zeros((1, *input_shape)) 
        
        x = F.relu(self.bn1(self.conv1(dummy_input)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.dropout1(self.conv2(x))))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        return x.numel() # numel() returns the total number of elements in the tensor


    def forward(self, x):
        # PyTorch Conv2D expects input in (batch_size, channels, height, width) format
        # If your x_train is (batch_size, height, width, channels), you'll need to permute it:
        # x = x.permute(0, 3, 1, 2) 

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.dropout1(self.conv2(x)))) # Dropout before BatchNormalization is common in Keras
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = torch.flatten(x, 1)  # Flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1) # Apply softmax on the output of the last linear layer

    def shared_step(self, batch, split='train'):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log(f'{split}_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
       return self.shared_step(batch, split='train')
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, split='val')
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        scheduler = CosineAnnealingLR(
            optimizer, T_max=10000, eta_min=1e-5
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()