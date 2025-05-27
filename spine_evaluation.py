import os
import time
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import cv2

from keypoint_spine_train import KeypointRegressionModel, UltrasoundKeypointDataset

TEST_DIR = "/images/test"
ANN_PATH = "/frame_annotations.json"


def evaluate(model, loader, line_thresh=25, iou_thresh=0.5):
    model.eval()
    classes = ["MID", "EPI", "VTB"]
    y_true = {c: [] for c in classes}
    y_pred = {c: [] for c in classes}

    t0 = time.perf_counter()
    with torch.no_grad():
        for imgs, labs in loader:
            raw = model(imgs).clamp(0, 1).cpu().numpy().reshape(-1, 3, 4)
            gt = labs.numpy().reshape(-1, 3, 4)

            for b in range(raw.shape[0]):
                for i, cls in enumerate(classes):
                    p = raw[b, i];
                    g = gt[b, i]

                    # detect
                    if cls in ("MID", "EPI"):
                        p_pts = (p.reshape(2, 2) * 224)
                        g_pts = (g.reshape(2, 2) * 224)
                        d1 = np.linalg.norm(p_pts[0] - g_pts[0])
                        d2 = np.linalg.norm(p_pts[1] - g_pts[1])
                        detected = (d1 < line_thresh and d2 < line_thresh)
                    else:
                        # IoU for VTB
                        xa, ya = p.reshape(2, 2).T
                        xb, yb = g.reshape(2, 2).T
                        xa_min, xa_max = xa.min(), xa.max()
                        ya_min, ya_max = ya.min(), ya.max()
                        xb_min, xb_max = xb.min(), xb.max()
                        yb_min, yb_max = yb.min(), yb.max()
                        xi1, yi1 = max(xa_min, xb_min), max(ya_min, yb_min)
                        xi2, yi2 = min(xa_max, xb_max), min(ya_max, yb_max)
                        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                        areaA = (xa_max - xa_min) * (ya_max - ya_min)
                        areaB = (xb_max - xb_min) * (yb_max - yb_min)
                        union = areaA + areaB - inter
                        iou = inter / union if union > 0 else 0
                        detected = (iou > iou_thresh)

                    has_gt = not ((g[:2] == 0).all() and (g[2:] == 0).all())

                    y_true[cls].append(int(has_gt))
                    y_pred[cls].append(int(detected))

    elapsed = time.perf_counter() - t0
    print(f"Eval took {elapsed:.2f}s over {sum(len(y_true[c]) for c in classes)} total eval instances")

    # per‐class
    p_list, r_list, f1_list, acc_list = [], [], [], []
    for cls in classes:
        p, r, f1, _ = precision_recall_fscore_support(
            y_true[cls], y_pred[cls],
            average="binary", zero_division=0
        )
        acc = accuracy_score(y_true[cls], y_pred[cls])
        print(f"{cls}: P={p:.2f}  R={r:.2f}  F1={f1:.2f}  Acc={acc:.2f}")
        p_list.append(p);
        r_list.append(r);
        f1_list.append(f1);
        acc_list.append(acc)

    # overall score
    p_avg = sum(p_list) / len(p_list)
    r_avg = sum(r_list) / len(r_list)
    f1_avg = sum(f1_list) / len(f1_list)
    acc_avg = sum(acc_list) / len(acc_list)
    print(f"Overall: P={p_avg:.2f}  R={r_avg:.2f}  F1={f1_avg:.2f}  Acc={acc_avg:.2f}")


def visualize_sample(model, image_path, transform, all_annots):
    model.eval()
    img_id = os.path.splitext(os.path.basename(image_path))[0]
    present = {label for (label, _, _) in all_annots.get(img_id, [])}

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    inp = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(inp).clamp(0, 1).cpu().numpy().reshape(-1, 2)
    pts = (out * np.array([w, h])).astype(int)

    im = np.array(img)
    names = ["MID", "EPI", "VTB"]
    for i, name in enumerate(names):
        if name not in present: continue
        p1 = tuple(pts[2 * i])
        p2 = tuple(pts[2 * i + 1])
        if name == "VTB":
            cv2.rectangle(im, p1, p2, (0, 255, 0), 1)
        else:
            cv2.line(im, p1, p2, (0, 255, 0), 1)
        cv2.putText(im, name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    plt.figure(figsize=(8, 6))
    plt.imshow(im)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved prediction visualization ➔ {image_path}")


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    with open(
            "/frame_annotations.json") as f:
        all_annots = json.load(f)

    device = torch.device("cpu")
    model = KeypointRegressionModel().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))

    imagenet_stats = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),

        transforms.Normalize(**imagenet_stats),
    ])
    test_ds = UltrasoundKeypointDataset(
        image_dir=TEST_DIR,
        annotation_path=ANN_PATH,
        transform=test_tf
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    evaluate(model, test_loader, line_thresh=25, iou_thresh=0.5)
    visualize_sample(model,
                     "/images/test/Image_93.png",
                     test_tf,
                     all_annots)
