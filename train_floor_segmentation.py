import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pycocotools.coco import COCO
import numpy as np
import cv2
import os
from floor_segmentation import FloorSegmentationModel

class COCOFloorDataset(Dataset):
    def __init__(self, coco_root, split='train'):
        self.coco_root = coco_root
        self.split = split
        
        # Load COCO annotations
        ann_file = os.path.join(coco_root, f'annotations/instances_{split}2017.json')
        self.coco = COCO(ann_file)
        
        # Get all images that have floor/ground annotations
        self.floor_cat_ids = self.coco.getCatIds(catNms=['floor', 'ground'])
        self.img_ids = []
        
        for cat_id in self.floor_cat_ids:
            self.img_ids.extend(self.coco.getImgIds(catIds=[cat_id]))
        self.img_ids = list(set(self.img_ids))
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        img_path = os.path.join(self.coco_root, f'{self.split}2017', img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create floor mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.floor_cat_ids)
        anns = self.coco.loadAnns(ann_ids)
        
        for ann in anns:
            mask = self.coco.annToMask(ann) | mask
        
        # Resize
        image = cv2.resize(image, (896, 896))
        mask = cv2.resize(mask, (896, 896))
        
        # Convert to tensor
        image = self.transform(image)
        mask = torch.from_numpy(mask).float() / 255.0
        
        return image, mask.unsqueeze(0)

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_floor_model.pth')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    coco_root = '/path/to/coco/dataset'  # Update this path
    train_dataset = COCOFloorDataset(coco_root, split='train')
    val_dataset = COCOFloorDataset(coco_root, split='val')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Initialize model
    model = FloorSegmentationModel()
    model = model.to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, num_epochs=10, device=device)

if __name__ == '__main__':
    main() 