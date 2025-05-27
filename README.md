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

