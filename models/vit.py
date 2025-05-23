from timm import create_model
from models.vision import VisionModel

class ViT(VisionModel):
    def __init__(
        self, model_name, freeze=False, dropout=0.2, model_kwargs={}
    ):
        super(ViT, self).__init__()
        self.save_hyperparameters()
        # input_shape is expected as (height, width, channels) from Keras, 
        # but PyTorch expects (channels, height, width) for Conv2d.
        
        # input -> B x C x H x W
        # model_kwargs = {'img_size': 64, 'patch_size': 8}
        self.backbone = create_model(
            model_name, pretrained=True, 
            num_classes=self.num_classes,
            drop_rate=dropout,
            in_chans=self.in_channels, **model_kwargs
        )
        if freeze:
        # Freeze all layers except classifier head
            for name, param in self.backbone.named_parameters():
                if not name.startswith("head."):
                    param.requires_grad = False