import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO
import time

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_ROOT = "./Object_detection"
IMG_DIR = "./Object_detection/images/100k/train"
LABEL_FILE = "./Object_detection/labels/100k/train/bdd100k_train_combined.json"
VAL_DIR = "./Object_detection/images/100k/val"
VAL_LABEL_FILE = "./Object_detection/labels/100k/val/bdd100k_val_combined.json"
Train_Size = 50000  #training images
Batch_Size = 32
CLASSES = ['car', 'person', 'traffic light'] 
CLASS_MAP = {'car': 0, 'person': 1, 'traffic light': 2}
INPUT_SIZE = 512
OUTPUT_SIZE = 128  # Stride 4 (512 / 4)

# ==========================================
# HELPER FUNCTIONS (Needed for Dataset)
# ==========================================
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    
    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    return heatmap

# ==========================================
# PART 1: DATASET HANDLING (DENSE TARGETS)
# ==========================================
class BDDDataset(Dataset):
    def __init__(self, img_dir, label_file, limit=Train_Size, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        print(f"Loading annotations from {label_file}...")
        
        if not os.path.exists(label_file):
            print(f"WARNING: Label file not found at {label_file}")
            self.data = []
            return

        with open(label_file) as f:
            full_data = json.load(f)
        
        self.data = []
        for entry in full_data:
            valid_objs = [x for x in entry.get('labels', []) if x['category'] in CLASSES]
            if valid_objs:
                self.data.append({'name': entry['name'], 'labels': valid_objs})
        
        if limit:
            self.data = self.data[:limit]
            print(f"Dataset truncated to {limit} images.")
        else:
            print(f"Loaded full dataset: {len(self.data)} images.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        info = self.data[idx]
        img_name = info['name']
        if not img_name.lower().endswith('.jpg'):
            img_name += ".jpg"

        img_path = os.path.join(self.img_dir, img_name)
        
        # 1. Load Image
        image = Image.open(img_path).convert("RGB")
        w_orig, h_orig = image.size
        image_resized = image.resize((INPUT_SIZE, INPUT_SIZE))
        
        if self.transform:
            image_tensor = self.transform(image_resized)
        else:
            image_tensor = T.ToTensor()(image_resized)

        # 2. Prepare Dense Targets (Fixed Size Tensors)
        # Heatmap: [3, 128, 128]
        heatmap = np.zeros((len(CLASSES), OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
        # Width/Height: [2, 128, 128]
        wh = np.zeros((2, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
        # Offset: [2, 128, 128] (To correct discretization error)
        reg_offset = np.zeros((2, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)
        # Mask: [128, 128] (Where do we actually have objects?)
        reg_mask = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)

        for obj in info['labels']:
            cat = obj['category']
            cls_id = CLASS_MAP[cat]
            b = obj['box2d']
            
            # Scale box to Input Size (512x512)
            x_scale = INPUT_SIZE / w_orig
            y_scale = INPUT_SIZE / h_orig
            x1 = b['x1'] * x_scale
            y1 = b['y1'] * y_scale
            x2 = b['x2'] * x_scale
            y2 = b['y2'] * y_scale
            
            # Convert to Output Size (128x128) logic
            # Calculate center and size in FEATURE SPACE
            w = (x2 - x1) / 4.0
            h = (y2 - y1) / 4.0
            cx = (x1 + x2) / 2.0 / 4.0
            cy = (y1 + y2) / 2.0 / 4.0
            
            # Integer center for grid placement
            cx_int, cy_int = int(cx), int(cy)
            
            if 0 <= cx_int < OUTPUT_SIZE and 0 <= cy_int < OUTPUT_SIZE:
                # 1. Paint Gaussian on Heatmap
                
                radius = max(2, int(min(w, h) / 3)) 
        
                draw_gaussian(heatmap[cls_id], (cx_int, cy_int), radius=radius)
                
                # 2. Store Width/Height at center pixel
                wh[0, cy_int, cx_int] = w
                wh[1, cy_int, cx_int] = h
                
                # 3. Store Offset (The difference between precise float center and int center)
                reg_offset[0, cy_int, cx_int] = cx - cx_int
                reg_offset[1, cy_int, cx_int] = cy - cy_int
                
                # 4. Mark this pixel as having an object
                reg_mask[cy_int, cx_int] = 1

        # Return Tensors directly
        return (
            image_tensor, 
            torch.from_numpy(heatmap), 
            torch.from_numpy(wh), 
            torch.from_numpy(reg_offset), 
            torch.from_numpy(reg_mask)
        )
# ==========================================
# PART 2: MODEL (Added Offset Head)
# ==========================================
class SimpleCenterNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.hm_head = nn.Conv2d(64, num_classes, 1)
        self.wh_head = nn.Conv2d(64, 2, 1)
        self.off_head = nn.Conv2d(64, 2, 1)  # NEW: Offset Head
        self.hm_head.bias.data.fill_(-2.19)
    def forward(self, x):
        features = self.backbone(x)
        features = self.upsample(features)
        
        hm = torch.sigmoid(self.hm_head(features))
        wh = self.wh_head(features)
        off = self.off_head(features)
        
        return hm, wh, off

# ==========================================
# PART 3: LOSS FUNCTION (Simplified)
# ==========================================
def focal_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_pos == 0: return -neg_loss
    return -(pos_loss + neg_loss) / num_pos

def dense_centernet_loss(pred_hm, pred_wh, pred_off, gt_hm, gt_wh, gt_off, mask):
    # 1. Heatmap Loss (Focal)
    hm_loss = focal_loss(pred_hm, gt_hm)
    
    # 2. Regression Loss (L1) - Only at object centers!
    # Expand mask to match dimensions
    mask_expanded = mask.unsqueeze(1).repeat(1, 2, 1, 1)
    
    wh_loss = F.l1_loss(pred_wh * mask_expanded, gt_wh * mask_expanded, reduction='sum')
    wh_loss = wh_loss / (mask.sum() + 1e-4)
    
    off_loss = F.l1_loss(pred_off * mask_expanded, gt_off * mask_expanded, reduction='sum')
    off_loss = off_loss / (mask.sum() + 1e-4)
    
    # Total Loss
    total_loss = (2.0 * hm_loss) + (0.1 * wh_loss) + (0.05 * off_loss)
    return total_loss, hm_loss, wh_loss, off_loss

# ==========================================
# PART 4: TRAINING LOOP
# ==========================================
def train_simple_model():
    print("\n--- Starting CenterNet Training ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ds = BDDDataset(IMG_DIR, LABEL_FILE, limit=Train_Size)
    
    loader = DataLoader(ds, batch_size=Batch_Size, shuffle=True, num_workers=2, pin_memory=True)
    
    model = SimpleCenterNet(num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    model.train()
    start_time = time.time()
    EPOCHS = 50
    for epoch in range(EPOCHS):
        total_loss = 0
        total_hm = 0
        total_wh = 0
        total_off = 0
        for imgs, gt_hm, gt_wh, gt_off, mask in loader:
            imgs = imgs.to(device)
            gt_hm = gt_hm.to(device)
            gt_wh = gt_wh.to(device)
            gt_off = gt_off.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            pred_hm, pred_wh, pred_off = model(imgs)
            
            loss, h_val, w_val, o_val = dense_centernet_loss(pred_hm, pred_wh, pred_off, 
                                                      gt_hm, gt_wh, gt_off, mask)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_hm += h_val.item()
            total_wh += w_val.item()
            total_off += o_val.item()        
        scheduler.step()
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f} | HM: {h_val.item():.4f} | WH: {w_val.item():.4f} | OFF: {o_val.item():.4f}")
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nCenterNet Training Finished!")
    print(f"Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average Time per Epoch: {total_time/EPOCHS:.2f} seconds")    
    torch.save(model.state_dict(), "simple_centernet.pth")
    return model

# ==========================================
# PART 5: DECODING (Updated for Dense Logic)
# ==========================================
def decode_detections(pred_hm, pred_wh, pred_off, threshold=0.3):
    batch_boxes = []
    stride = 4
    
    hm = F.max_pool2d(pred_hm, kernel_size=3, padding=1, stride=1)
    keep = (pred_hm == hm)
    pred_hm = pred_hm * keep.float()

    for i in range(pred_hm.shape[0]):
        img_boxes = []
        # Find peaks
        class_ids, rows, cols = torch.where(pred_hm[i] > threshold)
        
        for cls_id, r, c in zip(class_ids, rows, cols):
            score = pred_hm[i, cls_id, r, c].item()
            
            # 1. Get Width/Height (in feature pixels)
            w_feat = pred_wh[i, 0, r, c].item()
            h_feat = pred_wh[i, 1, r, c].item()
            
            # 2. Get Offset
            off_x = pred_off[i, 0, r, c].item()
            off_y = pred_off[i, 1, r, c].item()
            
            # 3. Calculate Center (Add integer index + decimal offset)
            cx_feat = c.item() + off_x
            cy_feat = r.item() + off_y
            
            # 4. Scale everything back to IMAGE SIZE
            
            w = w_feat * stride
            h = h_feat * stride
            cx = cx_feat * stride
            cy = cy_feat * stride
            
            x1, y1 = cx - w/2, cy - h/2
            x2, y2 = cx + w/2, cy + h/2
            
            img_boxes.append([x1, y1, x2, y2, score, cls_id.item()])
        
        batch_boxes.append(img_boxes)
    return batch_boxes

# ==========================================
# PART 6: VISUALIZATION
# ==========================================
def compare_models(simple_model, yolo_model, loader, device, num_images):
    print(f"\n--- Generating {num_images} Comparison Images ---")
    simple_model.eval()
    
    
    iter_loader = iter(loader)
    
    class_colors = {0: "blue", 1: "green", 2: "red"}
    try: font = ImageFont.truetype("arial.ttf", 16)
    except: font = ImageFont.load_default()

    for i in range(num_images):
        try:
            # Unpack the NEW tuple format
            batch = next(iter_loader)
            img_tensor = batch[0][0] # First image in batch
            
            img_pil_clean = T.ToPILImage()(img_tensor)
            img_pil_centernet = img_pil_clean.copy()
            img_pil_yolo = img_pil_clean.copy() 

            # CENTERNET INFERENCE
            start = time.time()
            with torch.no_grad():
                hm, wh, off = simple_model(img_tensor.unsqueeze(0).to(device))
                pred_boxes = decode_detections(hm, wh, off, threshold=0.3)[0]
            my_time = (time.time() - start) * 1000

            draw = ImageDraw.Draw(img_pil_centernet)
            for box in pred_boxes:
                x1, y1, x2, y2, score, cls = box
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(512, x2), min(512, y2)
                if x2 > x1 and y2 > y1:
                    color = class_colors.get(int(cls), "white")
                    label = f"{CLASSES[int(cls)]} {score:.2f}"
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    draw.text((x1, y1), label, fill="white", font=font)

            # YOLO INFERENCE
            start = time.time()
            yolo_res = yolo_model(img_pil_yolo, conf=0.5, verbose=False)[0] 
            yolo_time = (time.time() - start) * 1000
            yolo_plot = yolo_res.plot() 
            yolo_plot = cv2.cvtColor(yolo_plot, cv2.COLOR_BGR2RGB)

            print(f"Image {i+1}: CenterNet={my_time:.1f}ms | YOLO={yolo_time:.1f}ms")

            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            ax[0].imshow(img_pil_centernet)
            ax[0].set_title(f"CenterNet (Dense Targets)")
            ax[0].axis('off')
            ax[1].imshow(yolo_plot)
            ax[1].set_title(f"YOLOv8")
            ax[1].axis('off')
            plt.tight_layout()
            plt.savefig(f"test_final_result_{i}.jpg")
            plt.close(fig)

        except StopIteration:
            break

# ==========================================
# PART 7: YOLO HANDLING (Unchanged)
# ==========================================
def run_yolo():
    print("\n--- Training YOLOv8 (Reference) ---")
    mini_train_path = os.path.abspath("mini_train.txt")
    image_folder = IMG_DIR 
    if not os.path.exists(image_folder): return YOLO("yolov8n.pt") 

    all_images = sorted(os.listdir(image_folder))[:Train_Size] 
    with open(mini_train_path, "w") as f:
        for img in all_images:
            f.write(os.path.join(image_folder, img) + "\n")
            
    yaml_content = f"path: {os.path.abspath(DATASET_ROOT)}\ntrain: {mini_train_path}\nval: {mini_train_path}\nnames:\n  0: car\n  1: person\n  2: traffic light"
    with open("bdd_project.yaml", "w") as f: f.write(yaml_content)
        
    model = YOLO("yolov8n.pt") 
    model.train(data="bdd_project.yaml", epochs=50, imgsz=512, batch=Batch_Size, workers=8, project="bdd_yolo", name="run_comparison", exist_ok=True, verbose=False)
    return model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    # 1. TRAIN NEW CENTERNET
    simple_model = train_simple_model()
    
    # 2. TRAIN YOLO
    yolo_model = run_yolo()
    #yolo_model = YOLO(YOLO_PATH)
    # 3. COMPARE
    test_ds = BDDDataset(VAL_DIR, VAL_LABEL_FILE) # Small limit for viz
    loader = DataLoader(test_ds, batch_size=1, shuffle=True)
    compare_models(simple_model, yolo_model, loader, device, num_images=5)
