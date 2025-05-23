# VLASS-Vision

Vision model for VLASS images

| Model | Accuracy | F1-score | Pretrained | Frozen | Parameters | Augmentation |
|:--:|:--:|:--:|:--:|:--:|:--:|
| CNN | 83.9 | 75.5 | False | False | 867K | No |
| CNN | 83.9 | 75.5 | False | False | 867K | Yes |
| Resnet18 | 73.8 | 60.1 | True | True | 11.2M | No |
| ResNet18 | 88.1 | 81.4 | True | False | 11.2M | No | 
| ResNet18 | 90.2 | 84.5 | True | False | 11.2M | Yes |
| ViT-Tiny | 84.0 | 73.3 | True | True | 5.4M | No |
| ViT-Tiny | 91.5 | 86.9 | True | False | 5.4M | No |
| ViT-Tiny | 92.5 | 88.3 | True | False | 5.4M | Yes |