import os
import json
import math

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


IMAGE_DIR = "C:/Users/Malar/Downloads/OneDrive_2025-04-16/HiCura Medical Take Home Dataset/images/train"
VAL_DIR = "C:/Users/Malar/Downloads/OneDrive_2025-04-16/HiCura Medical Take Home Dataset/images/train/val"
ANN_PATH = "C:/Users/Malar/Downloads/OneDrive_2025-04-16/HiCura Medical Take Home Dataset/frame_annotations.json"

# —————— Load annotations & compute priors ——————
with open(ANN_PATH) as f:
    ann = json.load(f)

mid_lens, epi_lens = [], []
ymid_centers, xepi_ends = [], []
yvtb_centers = []
mid_angles = []

for img_id, shapes in ann.items():
    for base in (IMAGE_DIR, VAL_DIR):
        img_file = os.path.join(base, f"{img_id}.png")
        if os.path.exists(img_file):
            w, h = Image.open(img_file).size
            for label, kind, pts in shapes:
                (x1, y1), (x2, y2) = pts
                nx1, ny1 = x1 / w, y1 / h
                nx2, ny2 = x2 / w, y2 / h

                if label == "MID":
                    # length & center
                    mid_lens.append(np.hypot(nx2 - nx1, ny2 - ny1))
                    ymid_centers.append((ny1 + ny2) / 2)
                    # true angle
                    mid_angles.append(math.atan2(ny2 - ny1, nx2 - nx1))

                elif label == "EPI":
                    epi_lens.append(np.hypot(nx2 - nx1, ny2 - ny1))
                    xepi_ends.append(nx2)

                elif label == "VTB":
                    yvtb_centers.append((ny1 + ny2) / 2)
            break


def circular_median(angles):

    angles = np.asarray(angles)
    best = angles[0]
    min_cost = np.inf

    for i in angles:
        diffs = np.angle(np.exp(1j * (angles - i)))
        cost = np.sum(np.abs(diffs))
        if cost < min_cost:
            min_cost = cost
            best = i

    return best


# compute priors
mid_mean = np.mean(mid_lens)
epi_mean = np.mean(epi_lens)
ymid_mean = np.mean(ymid_centers)
xepi_mean = np.mean(xepi_ends)
yvtb_mean = np.mean(yvtb_centers)
angles = np.array(mid_angles)
median_angle = circular_median(angles)


# —————— Dataset ——————
class UltrasoundKeypointDataset(Dataset):
    def __init__(self, image_dir, annotation_path, transform):
        self.image_dir = image_dir
        self.transform = transform
        with open(annotation_path) as f:
            self.annotations = json.load(f)

        self.images = sorted(
            f for f in os.listdir(image_dir)
            if f.endswith(".png") and f[:-4] in self.annotations
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fn = self.images[idx]
        img = Image.open(os.path.join(self.image_dir, fn)).convert("RGB")
        w, h = img.size

        coords = {"MID": [0, 0, 0, 0], "EPI": [0, 0, 0, 0], "VTB": [0, 0, 0, 0]}
        for label, _, pts in self.annotations[fn[:-4]]:
            (x1, y1), (x2, y2) = pts
            coords[label] = [x1 / w, y1 / h, x2 / w, y2 / h]

        return (
            self.transform(img),
            torch.tensor(coords["MID"] + coords["EPI"] + coords["VTB"],
                         dtype=torch.float32)
        )


# —————— Model & Composite Loss  ——————
class KeypointRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 12)
        )
        self.net = backbone

    def forward(self, x):
        return self.net(x)


class CompositeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # one log-var per task: {MSE, angle, length, mid-center, epi-endpoint, vtb-center}
        self.log_vars = nn.Parameter(torch.zeros(6))

    def forward(self, pred, tgt):
        B = pred.size(0)
        p = pred.view(B, 6, 2)
        t = tgt.view(B, 6, 2)
        # mask[i,j]=1 if structure j present in sample i
        mask = (t.abs().sum(dim=2) > 0).float()  # (B,6)

        # —— Task 0: masked MSE ——
        se = (pred - tgt).pow(2).view(B, 6, 2).sum(dim=2)  # (B,6)
        denom = mask.sum().clamp(min=1.0)
        L0 = (se * mask).sum() / denom

        # —— Task 1: angle priors  ——
        dxm = p[:, 1, 0] - p[:, 0, 0];
        dym = p[:, 1, 1] - p[:, 0, 1]
        Lmid = ((torch.atan2(dym, dxm) - median_angle).pow(2) * mask[:, 0]).sum() \
               / mask[:, 0].sum().clamp(min=1.0)

        dxe = p[:, 3, 0] - p[:, 2, 0];
        dye = p[:, 3, 1] - p[:, 2, 1]
        Lepi = ((torch.atan2(dye, dxe) - 0.0).pow(2) * mask[:, 1]).sum() \
               / mask[:, 1].sum().clamp(min=1.0)

        L1 = Lmid + Lepi

        # —— Task 2: length priors ——
        lm = torch.norm(p[:, 1] - p[:, 0], dim=1)
        le = torch.norm(p[:, 3] - p[:, 2], dim=1)
        L2 = (((lm - mid_mean).pow(2) * mask[:, 0]).sum() / mask[:, 0].sum().clamp(min=1.0)
              + ((le - epi_mean).pow(2) * mask[:, 1]).sum() / mask[:, 1].sum().clamp(min=1.0))

        # —— Task 3: MID center-bias ——
        cY = (p[:, 0, 1] + p[:, 1, 1]) * 0.5
        L3 = (((cY - ymid_mean).pow(2) * mask[:, 0]).sum() / mask[:, 0].sum().clamp(min=1.0))

        # —— Task 4: EPI endpoint bias ——
        eX = p[:, 3, 0]
        L4 = ((F.relu(xepi_mean - eX).pow(2) * mask[:, 1]).sum()
              / mask[:, 1].sum().clamp(min=1.0))

        # —— Task 5: VTB center-bias ——
        vY = (p[:, 4, 1] + p[:, 5, 1]) * 0.5
        L5 = (((vY - yvtb_mean).pow(2) * mask[:, 2]).sum()
              / mask[:, 2].sum().clamp(min=1.0))

        # —— combine via learned uncertainty weights ——
        losses = [L0, L1, L2, L3, L4, L5]
        total = 0
        for i, Li in enumerate(losses):
            total = total + (Li * torch.exp(-self.log_vars[i]) + self.log_vars[i])

        return total


# —————— Training Loop ——————
def train():
    imagenet_stats = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(0.2, 0.2),
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.15)),
        transforms.ToTensor(),

        transforms.Normalize(**imagenet_stats),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),

        transforms.Normalize(**imagenet_stats),
    ])

    train_ds = UltrasoundKeypointDataset(IMAGE_DIR, ANN_PATH, train_tf)
    val_ds = UltrasoundKeypointDataset(VAL_DIR, ANN_PATH, val_tf)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeypointRegressionModel().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    lossf = CompositeLoss().to(device)

    best_val = float("inf")
    for ep in range(1, 151):
        model.train()
        running = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = lossf(preds, labels)
            opt.zero_grad();
            loss.backward();
            opt.step()
            running += loss.item()

        print(f"Epoch {ep:3d} train_loss = {running / len(train_loader):.4f}")

        model.eval()
        vloss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                vloss += lossf(model(imgs), labels).item()
        vloss /= len(val_loader)
        print(f"val_loss   = {vloss:.4f}")

        if vloss < best_val:
            best_val = vloss
            torch.save(model.state_dict(), "best_model.pth")
            print("saved new best model!")


if __name__ == "__main__":
    train()


