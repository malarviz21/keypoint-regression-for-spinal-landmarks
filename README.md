**Spinal Keypoint Regression on Ultrasound Images**


**Objective**: Build a lightweight and fast DL model that directly predicts the locations of three spine landmarks (midline, epidural space, and vertebral body) from ultrasound images.

**Clinical Motivation :**
Problem: Accurate placement of neuraxial block needles using spinal ultrasound is challenging.
Solution: Real-time DL assistance to highlight key anatomic landmarks.
Impact: 
    Reduces procedure time and lowers complication rates.
    Reduces needle misplacement risk
    Aids training/decision support
Real-time constraints:
    Decision support must appear instantly (<100 ms)
    Regression model chosen for timing requirements.

<img width="334" alt="image" src="https://github.com/user-attachments/assets/645271ad-be81-4a17-9c4a-0be5ca8a8403" />




**Dataset and Annotations:**
Data Collection: Transverse-view ultrasound frames in .png image format.
Split: 80% train with validation, 20% test
Annotations: frame_annotations.json contains:
MID & EPI as line endpoints
VTB as rectangle corners
Format: [["MID","line",[[x1,y1],[x2,y2]]], …]
"Image_41": [
  ["VTB","rect",[[258,349],[399,384]]],
  ["MID","line",[[330,88],[331,388]]],
  ["EPI","line",[[272,224],[381,224]]]
]
