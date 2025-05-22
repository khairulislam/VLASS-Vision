import lightning as L
from timm import create_model
from models.vision import VisionModel

class ViT(VisionModel):
    def __init__(self, num_classes):
        super(VisionModel, self).__init__()
        self.save_hyperparameters()
        # input_shape is expected as (height, width, channels) from Keras, 
        # but PyTorch expects (channels, height, width) for Conv2d.
        
        # input -> B x C x H x W
        model_kwargs = {'img_size': 64, 'patch_size': 8}
        self.backbone = create_model(
            'vit_tiny_patch16_224', pretrained=True, num_classes=num_classes,
            in_chans=1, **model_kwargs
        )