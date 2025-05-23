import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import CosineAnnealingLR

class VisionModel(L.LightningModule):
    def __init__(self):
        super(VisionModel, self).__init__()
        
        self.num_classes = 4
        self.in_channels = 1
        self.save_hyperparameters()
        # input_shape is expected as (height, width, channels) from Keras, 
        # but PyTorch expects (channels, height, width) for Conv2d.
        
        # input -> B x C x H x W
        self.backbone = None
        
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    
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
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        scheduler = CosineAnnealingLR(
            optimizer, T_max=10000, eta_min=1e-6
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()