# Final Test Evaluation Report: 415 Held-Out PCB Images Benchmark

This report presents the final evaluation of the 4 classical computer vision defect inspection algorithms on the **415 Held-Out Test PCB Images** (`data/dataset_split.csv`) using the frozen optimal parameters selected during validation.

All evaluations are conducted against ground truth Pascal VOC annotations using the standard benchmark criterion ($\text{IoU} \ge 0.50$).

---

## 1. Executive Summary: Test Set Leaderboard (415 Unseen Images)

| Rank | Algorithm | Frozen Optimal Configuration | Test Precision | Test Recall | **Test F1-Score** | Mean IoU | Mean Latency | FPS | Generalization Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 **1** | **Enhanced Otsu** | `MedianBlur=3, Dilate=35, MinArea=150` | **84.40%** | **81.50%** | **`0.8293`** | **0.6015** | **22.3 ms** | **44.9** | **Winner (No Overfitting)** |
| 🥈 **2** | **Enhanced Template Matching** | `Block=32x32, Step=16, Thresh=0.65` | **38.10%** | **57.13%** | **`0.4571`** | **0.5213** | **433.2 ms** | **2.3** | Stable Secondary Inspector |
| 🥉 **3** | **Enhanced Canny** | `Low=50, High=150, Close=5, MinArea=300` | **3.87%** | **9.64%** | **`0.0553`** | **0.1826** | **70.4 ms** | **14.2** | Fragmented Edge Boxes |
| 🎖️ **4** | **Enhanced ORB** | `Hamming=40, R=20, Merge=True, MinArea=300` | **0.33%** | **12.58%** | **`0.0064`** | **0.2082** | **564.5 ms** | **1.8** | Textureless False Positives |

---

## 2. Validation (139 Images) vs. Test (415 Images) Generalization Analysis

| Algorithm | Validation F1 (139 Images) | Test F1 (415 Images) | $\Delta$ F1 (%) | Generalization Finding |
| :--- | :---: | :---: | :---: | :--- |
| **Enhanced Otsu** | **83.96%** | **82.93%** | **-1.03%** | **Excellent generalization**: The 35×35 morphological dilation and median filter scale perfectly to unseen boards without hyperparameter decay. |
| **Enhanced Template Matching** | **45.02%** | **45.71%** | **+0.69%** | **Highly consistent**: Captures over 57% of ground truth defects across all unseen test boards. |
| **Enhanced Canny** | **5.28%** | **5.53%** | **+0.25%** | **Consistently low**: 1-pixel boundary differences consistently fail to meet IoU $\ge 0.50$ without extensive solid region dilation. |
| **Enhanced ORB** | **0.61%** | **0.64%** | **+0.03%** | **Theoretically constrained**: FAST feature extraction consistently fails on uniform copper traces and smooth solder masks. |

---

## 3. Per-Defect Category Breakdown on Test Set (Enhanced Otsu)

Across the 415 test images, Enhanced Otsu demonstrates exceptional defect coverage across 5 of the 6 defect categories:

| Defect Category | Test Images | True Positives | False Positives | False Negatives | Precision | Recall | **F1-Score** | Mean IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Missing Hole** | 69 | 283 | 30 | 16 | **90.42%** | **94.65%** | **`0.9248`** | **0.7528** |
| **Mouse Bite** | 69 | 259 | 19 | 36 | **93.17%** | **87.80%** | **`0.9040`** | **0.6218** |
| **Spurious Copper** | 70 | 274 | 42 | 35 | **86.71%** | **88.67%** | **`0.8768`** | **0.6577** |
| **Spur** | 69 | 248 | 31 | 47 | **88.89%** | **84.07%** | **`0.8641`** | **0.5480** |
| **Open Circuit** | 69 | 232 | 38 | 57 | **85.93%** | **80.28%** | **`0.8301`** | **0.5952** |
| **Short Circuit** | 69 | 149 | 107 | 137 | **58.20%** | **52.10%** | **`0.5498`** | **0.4327** |

---

## 4. Key Insights for Assignment Report

1. **Why Missing Hole and Mouse Bite achieve >90% F1**:
   - Circular holes and trace borders produce strong, contiguous intensity differences that perfectly match bounding box morphology after 35×35 dilation.
2. **Why Short Circuit is More Challenging (55.0% F1)**:
   - Hairline short-circuit bridges between dense parallel pins sometimes produce narrow difference segments that require higher adaptive dilation or multi-scale morphological closing.
3. **Execution Artifacts**:
   - Notebook: `notebooks/final_evaluation/06_final_evaluation.ipynb`
   - Results CSV: `outputs/metrics/final_test_results.csv`
   - Summary JSON: `outputs/metrics/final_test_summary.json`
