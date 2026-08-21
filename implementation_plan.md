# Code Improvements: Preprocessing, ORB Calibration, and Dashboard Enhancement

## Summary

Four code-level improvements to address rubric-critical gaps identified in the review:
1. Add a **preprocessing module** (noise filtering + contrast enhancement) applied uniformly to all four algorithms
2. Add **spatial calibration** to the ORB algorithm (homography rectification + pixel-to-mm scaling)
3. Provide **ORB pseudocode** (plain text) matching the updated code
4. Upgrade **Streamlit dashboard** with Defect Property Summary Table, Automated PDF Export, and Batch/Folder Inspection Mode

No hybrid/enhanced approach will be implemented at this stage. The four algorithms remain as baselines with shared preprocessing added fairly to all.

---

## User Review Required

> [!IMPORTANT]
> **Frozen parameters will need updating.** Adding preprocessing changes the input to all four algorithms, so `frozen_parameters.yaml` needs a new section for preprocessing settings (Gaussian kernel size, CLAHE clip limit, etc.). The existing test results in `final_test_summary.json` were generated without preprocessing, so a re-evaluation on the test split will be needed after preprocessing is added, or the old results should be kept as "Phase 1 (Raw)" and new results recorded as "Phase 2 (With Preprocessing)".

> [!IMPORTANT]
> **ORB code will change significantly.** The current ORB detects defects via BF cross-check + spatial displacement + unmatched keypoints → defect boxes. Adding calibration means ORB will now also perform KNN + Lowe ratio + RANSAC homography → `warpPerspective` to align the test image *before* defect detection. This is a functional change to `orb.py`. The existing ORB frozen parameters and test metrics will need re-evaluation.

> [!WARNING]
> **PDF export requires `fpdf2` or `reportlab`.** Neither is currently in `requirements.txt`. I will use `fpdf2` (lightweight, pure Python) to generate inspection certificate PDFs.

## Open Questions

> [!IMPORTANT]
> **Q1: Phase 1 vs Phase 2 evaluation strategy.** Should the existing raw baseline results be preserved as "Phase 1" and the preprocessed results recorded as a separate "Phase 2" comparison? Or should preprocessing simply replace the current raw pipeline and the old results be discarded?

> [!IMPORTANT]
> **Q2: ORB calibration scope.** Should ORB calibration (homography alignment) be applied:
> - **(A)** Only within the ORB algorithm itself (ORB aligns then detects), or
> - **(B)** As a shared pre-alignment step available to all four algorithms?
>
> Option A keeps the comparison fair (only ORB gets alignment because it naturally produces a homography). Option B means all algorithms benefit from alignment equally.

> [!IMPORTANT]
> **Q3: Pixel-to-mm calibration constant.** The PCB images are 600×600 pixels. Based on standard PCB via hole diameters (~0.8 mm), one pixel ≈ 0.05–0.10 mm. Should I estimate this from the dataset's drill hole annotations, or do you have a known physical board dimension to use?

---

## Proposed Changes

### 1. Preprocessing Module

#### [NEW] [`algorithms/preprocessing.py`](file:///c:/Users/jieer/Documents/GitHub/pcb-defects-inspection/algorithms/preprocessing.py)

A new module providing two preprocessing functions applied uniformly to all algorithms:

```python
def denoise(gray: np.ndarray, method: str = "gaussian", kernel_size: int = 3) -> np.ndarray:
    """Apply noise suppression filter.
    
    Methods:
    - "gaussian": cv2.GaussianBlur with sigma=0 (auto-calculated)
    - "median": cv2.medianBlur for impulse noise
    """

def enhance_contrast(gray: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation)."""

def preprocess_image(gray: np.ndarray, config: dict) -> np.ndarray:
    """Apply the full preprocessing pipeline (denoise → enhance) based on config."""
```

#### [MODIFY] [`algorithms/common.py`](file:///c:/Users/jieer/Documents/GitHub/pcb-defects-inspection/algorithms/common.py)

Update `preprocess_pair()` to optionally apply the preprocessing pipeline:

```python
def preprocess_pair(
    reference: np.ndarray,
    defective: np.ndarray,
    preprocessing_config: dict | None = None,  # NEW parameter
) -> tuple[np.ndarray, np.ndarray]:
```

When `preprocessing_config` is `None` (default), behaviour is unchanged (backward compatible). When provided, both images pass through `denoise()` → `enhance_contrast()` after grayscale conversion.

#### [MODIFY] [`configs/frozen_parameters.yaml`](file:///c:/Users/jieer/Documents/GitHub/pcb-defects-inspection/configs/frozen_parameters.yaml)

Add preprocessing section:

```yaml
preprocessing:
  enabled: true
  denoise_method: gaussian    # "gaussian" | "median" | "none"
  denoise_kernel_size: 3      # 3 or 5
  contrast_enhancement: clahe # "clahe" | "none"
  clahe_clip_limit: 2.0
  clahe_tile_size: 8
```

#### [MODIFY] Each algorithm file

Each of [`otsu.py`](file:///c:/Users/jieer/Documents/GitHub/pcb-defects-inspection/algorithms/otsu.py), [`canny.py`](file:///c:/Users/jieer/Documents/GitHub/pcb-defects-inspection/algorithms/canny.py), [`template_matching.py`](file:///c:/Users/jieer/Documents/GitHub/pcb-defects-inspection/algorithms/template_matching.py), [`orb.py`](file:///c:/Users/jieer/Documents/GitHub/pcb-defects-inspection/algorithms/orb.py) will gain an optional `preprocessing_config` parameter in their `detect_*()` function, passed through to `preprocess_pair()`. Default is `None` for full backward compatibility.

---

### 2. ORB Spatial Calibration & Rectification

#### [MODIFY] [`algorithms/orb.py`](file:///c:/Users/jieer/Documents/GitHub/pcb-defects-inspection/algorithms/orb.py)

Add a new function and update the detection flow:

```python
def calibrate_alignment(
    reference_gray: np.ndarray,
    defective_gray: np.ndarray,
    n_features: int = 5000,
    scale_factor: float = 1.2,
    n_levels: int = 8,
    ratio_threshold: float = 0.75,
    ransac_reproj_threshold: float = 5.0,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Compute ORB homography and warp defective image to align with reference.
    
    Returns:
        aligned_defective: The warped defective image (or original if alignment fails)
        homography_matrix: 3×3 homography (or None if insufficient matches)
        inlier_count: Number of RANSAC inliers
    """
```

The existing `detect_orb()` function will gain an optional `calibrate: bool = False` parameter. When `True`:
1. First run `calibrate_alignment()` to warp the defective image
2. Then run the existing BF cross-check defect detection on the aligned pair

The `ORBDetection` dataclass will gain new fields:
- `homography_matrix: np.ndarray | None`
- `inlier_count: int`
- `calibrated: bool`

#### [MODIFY] [`configs/frozen_parameters.yaml`](file:///c:/Users/jieer/Documents/GitHub/pcb-defects-inspection/configs/frozen_parameters.yaml)

Add to the `orb` section:

```yaml
orb:
  calibrate: true
  ransac_reproj_threshold: 5.0
  # ... existing parameters unchanged
```

#### ORB Pseudocode (Plain Text)

```
Algorithm 3: ORB Feature Detection with Spatial Calibration for PCB Defect Detection

Input:  Reference PCB Image (R), Test PCB Image (T),
        Max Features (N = 5000), Scale Factor (sf = 1.2), Pyramid Levels (L = 8),
        Ratio Threshold (r = 0.75), RANSAC Reprojection Threshold (rr = 5.0),
        Spatial Distance Threshold (d_thresh = 15.0),
        Hamming Distance Threshold (h_thresh = 60.0),
        Box Radius (br = 35)
Output: Defect Bounding Boxes (L_boxes), Homography Matrix (H)

Begin
--- Stage 1: Spatial Calibration ---
1:  R_gray <- ToGrayscale(R),    T_gray <- ToGrayscale(T)
2:  detector <- ORB_Create(nfeatures = N, scaleFactor = sf, nlevels = L)
3:  (K_R, D_R) <- detector.DetectAndCompute(R_gray)
4:  (K_T, D_T) <- detector.DetectAndCompute(T_gray)
5:  matcher <- BFMatcher(normType = NORM_HAMMING, crossCheck = False)
6:  knn_matches <- matcher.KnnMatch(D_T, D_R, k = 2)
7:  M_good <- empty list
8:  for each pair (m, n) in knn_matches do
9:      if distance(m) < r * distance(n) then    (Lowe's ratio test)
10:         M_good <- M_good + {m}
11:     end if
12: end for
13: P_T <- coordinates of matched keypoints from K_T
14: P_R <- coordinates of matched keypoints from K_R
15: H, inlier_mask <- FindHomography(P_T, P_R, method = RANSAC, threshold = rr)
16: T_aligned <- WarpPerspective(T_gray, H, dsize = SizeOf(R_gray))

--- Stage 2: Defect Detection via Feature Anomaly ---
17: (K_R2, D_R2) <- detector.DetectAndCompute(R_gray)
18: (K_A2, D_A2) <- detector.DetectAndCompute(T_aligned)
19: matcher2 <- BFMatcher(normType = NORM_HAMMING, crossCheck = True)
20: matches <- matcher2.Match(D_R2, D_A2)
21: defect_points <- empty list
22: matched_indices <- empty set
23: for each match m in matches do
24:     matched_indices <- matched_indices + {m.trainIdx}
25:     pt_ref <- K_R2[m.queryIdx].pt
26:     pt_def <- K_A2[m.trainIdx].pt
27:     disp <- EuclideanDistance(pt_def, pt_ref)
28:     if disp > d_thresh or m.distance > h_thresh then
29:         defect_points <- defect_points + {pt_def}    (Inconsistent match)
30:     end if
31: end for
32: for each keypoint kp at index idx in K_A2 do
33:     if idx not in matched_indices then
34:         defect_points <- defect_points + {kp.pt}    (Unmatched keypoint)
35:     end if
36: end for
37: L_boxes <- empty list
38: for each point (x, y) in defect_points do
39:     box <- [max(0, x-br), max(0, y-br), min(W, x+br), min(H_img, y+br)]
40:     L_boxes <- L_boxes + {box}
41: end for
42: return Defect Bounding Boxes L_boxes, Homography Matrix H
End
```

---

### 3. Dashboard Enhancements

#### [MODIFY] [`app.py`](file:///c:/Users/jieer/Documents/GitHub/pcb-defects-inspection/app.py)

Major restructure of the Streamlit dashboard into three sections:

**3A. Defect Property Summary Table**

After detection, display an interactive `st.dataframe` table with columns:
| Defect # | x_min | y_min | x_max | y_max | Width (px) | Height (px) | Area (px²) | Aspect Ratio | Similarity Score |
|---|---|---|---|---|---|---|---|---|---|

If pixel-to-mm calibration is enabled, additional columns: `Width (mm)`, `Height (mm)`, `Area (mm²)`.

**3B. Automated PDF Inspection Certificate Export**

New dependency: `fpdf2` (add to `requirements.txt`).

A `st.download_button("Download Inspection Certificate (PDF)")` that generates a PDF containing:
- Header: "PCB Defect Inspection Certificate"
- Timestamp, image dimensions, algorithm used, preprocessing settings
- Summary metrics (total candidates, processing time)
- The defect property table
- The overlay image (reference, defective, detection overlay side-by-side)
- Footer with pass/fail status (if defect count > 0, status = "DEFECTS DETECTED")

**3C. Batch / Folder Inspection Mode**

Add a new tab or section using `st.tabs(["Single Inspection", "Batch Inspection"])`:
- **Single Inspection**: Current upload-two-images flow (unchanged).
- **Batch Inspection**: 
  - User selects a folder from the test split via dropdown (or uploads multiple image pairs)
  - Runs detection on all pairs sequentially
  - Displays aggregate summary table with per-image defect counts, processing times
  - Provides a "Download All Results (JSON)" button

#### [MODIFY] [`requirements.txt`](file:///c:/Users/jieer/Documents/GitHub/pcb-defects-inspection/requirements.txt)

Add `fpdf2` for PDF generation.

---

## File Change Summary

| Action | File | Description |
|--------|------|-------------|
| NEW | `algorithms/preprocessing.py` | Gaussian/Median denoising + CLAHE contrast enhancement |
| MODIFY | `algorithms/common.py` | Add optional `preprocessing_config` parameter to `preprocess_pair()` |
| MODIFY | `algorithms/otsu.py` | Pass preprocessing config through to `preprocess_pair()` |
| MODIFY | `algorithms/canny.py` | Pass preprocessing config through to `preprocess_pair()` |
| MODIFY | `algorithms/template_matching.py` | Pass preprocessing config through to `preprocess_pair()` |
| MODIFY | `algorithms/orb.py` | Add `calibrate_alignment()` function + `calibrate` flag in `detect_orb()` |
| MODIFY | `configs/frozen_parameters.yaml` | Add `preprocessing` section and `orb.calibrate` flag |
| MODIFY | `app.py` | Add Property Table, PDF Export, Batch Mode |
| MODIFY | `requirements.txt` | Add `fpdf2` |

---

## Verification Plan

### Automated Tests

```bash
# Run existing test suite to confirm backward compatibility
python -m pytest tests/ -v

# Quick smoke test: run one image through each algorithm with preprocessing
python -c "
from algorithms.common import load_image, preprocess_pair
from algorithms.otsu import detect_otsu
ref = load_image('data/raw/PCB_DATASET/PCB_USED/01.JPG')
defective = load_image('data/raw/PCB_DATASET/images/Missing_hole/01_missing_hole_01.jpg')
result = detect_otsu(ref, defective, preprocessing_config={'denoise_method': 'gaussian', 'denoise_kernel_size': 3, 'contrast_enhancement': 'clahe', 'clahe_clip_limit': 2.0, 'clahe_tile_size': 8})
print(f'Otsu with preprocessing: {len(result.boxes)} boxes, {result.processing_time_ms:.1f} ms')
"

# Test ORB calibration
python -c "
from algorithms.common import load_image
from algorithms.orb import detect_orb
ref = load_image('data/raw/PCB_DATASET/PCB_USED/01.JPG')
defective = load_image('data/raw/PCB_DATASET/images/Missing_hole/01_missing_hole_01.jpg')
result = detect_orb(ref, defective, calibrate=True)
print(f'ORB calibrated: {result.calibrated}, inliers: {result.inlier_count}, boxes: {len(result.boxes)}')
"
```

### Manual Verification

- Launch `streamlit run app.py` and verify:
  - Property summary table displays correctly
  - PDF export downloads and opens properly
  - Batch mode processes multiple images and shows aggregate results
