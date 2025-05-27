# VLASS-Vision

Vision model for VLASS images

| Model | Accuracy | F1-score | Pretrained | Frozen | Parameters | Augmentation |
|:--:|:--:|:--:|:--:|:--:|:--:|
| CNN | 83.9 | 75.5 | False | False | 867K | No |
| CNN | 83.9 | 75.5 | False | False | 867K | Yes |
| Resnet18 | 73.8 | 60.1 | True | True | 11.2M | No |
| ResNet18 | 88.1 | 81.4 | True | False | 11.2M | No | 
| ResNet18 | 90.2 | 84.5 | True | False | 11.2M | Yes |
| MobileNet | 88.6 | 81.9 | True | False | 2.2M | Yes |
| EfficientNet | 83.4 | 74.9 | True | False | 10.7M | Yes | 
| ViT-Tiny | 84.0 | 73.3 | True | True | 5.4M | No |
| ViT-Tiny | 91.5 | 86.9 | True | False | 5.4M | No |
<!-- | ViT-Tiny | 92.5 | 88.3 | True | False | 5.4M | Yes | -->
| ViT-Small | | | True | False | 21.3M | Yes |
| ViT-Base | 91.9 | 87.9 | True | False | 85.2M | No |
| ViT-Base | 92.5 | 88.3 | True | False | 85.2M | Yes |
| SwinViT | 89.3 | 83.1 | True | False | 70.1M | No |
| SwinViT | 89.6 | 83.5 | True | False | 70.1M | Yes |