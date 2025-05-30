# Floor Segmentation using DINOv2

This project implements floor segmentation using DINOv2 as the backbone and a custom decoder head. The model is trained on the COCO dataset to detect floors and ground surfaces in images.

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Training

1. Download the COCO dataset and update the path in `train_floor_segmentation.py`:
```python
coco_root = '/path/to/coco/dataset'  # Update this path
```

2. Run the training script:
```bash
python train_floor_segmentation.py
```

The best model will be saved as `best_floor_model.pth`.

## Inference

To detect floors in an image:

```python
from floor_segmentation import detect_floor

# Detect floor in an image
floor_mask, visualization = detect_floor(
    'path/to/image.jpg',
    model_path='best_floor_model.pth'
)

# Save results
cv2.imwrite('floor_mask.png', floor_mask)
cv2.imwrite('floor_visualization.png', visualization)
```

## Model Architecture

The model uses:
- DINOv2 ViT-L/14 as the backbone
- Feature Pyramid Network (FPN) for multi-scale feature extraction
- Custom decoder head for floor segmentation

## Dataset

The model is trained on the COCO dataset, specifically using images that contain floor/ground annotations. The dataset is automatically filtered to include only relevant images during training.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 