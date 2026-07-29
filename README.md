# PCB Defect Inspection Using Image Processing Techniques

## Project Overview

This project investigates and compares four image processing techniques for automated bare printed circuit board (PCB) defect inspection:

1. Otsu's Thresholding
2. Canny Edge Detection
3. Template Matching
4. ORB Feature Matching

The system focuses on six common bare-PCB fabrication defects:

- Open circuit
- Short circuit
- Spur
- Spurious copper
- Missing hole
- Mouse bite

All techniques will be implemented and evaluated under the same experimental conditions. The comparison will consider detection performance, localisation quality, robustness, and processing time. Based on the results, an enhanced or hybrid method may be developed to improve overall inspection performance.

## Project Objectives

- Investigate contemporary image processing techniques for PCB defect inspection.
- Implement four different techniques using a common dataset and evaluation procedure.
- Compare the accuracy, robustness, localisation capability, and computational efficiency of each technique.
- Identify the strengths and limitations of each technique.
- Develop an enhanced or hybrid method based on the experimental findings.
- Integrate the final methods into a simple Streamlit dashboard for live testing.

## Techniques

### 1. Otsu's Thresholding

Otsu's method automatically selects a global threshold to separate foreground and background pixels. In this project, it may be applied to the difference between a defect-free reference image and a test PCB image to produce a binary defect mask.

### 2. Canny Edge Detection

Canny edge detection extracts the structural boundaries of PCB tracks and holes. Differences between the reference and test edge maps can be analysed to identify structural defects.

### 3. Template Matching

Template matching compares the visual similarity between a test PCB image and a corresponding defect-free reference image or image region.

### 4. ORB Feature Matching

ORB, or Oriented FAST and Rotated BRIEF, detects keypoints and generates binary descriptors. Feature correspondences between reference and test images can be analysed to identify structural differences and support image alignment.

## Recommended Project Structure

```text
pcb-defects-inspection/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── algorithms/
│   ├── __init__.py
│   ├── common.py
│   ├── otsu.py
│   ├── canny.py
│   ├── template_matching.py
│   └── orb.py
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_otsu.ipynb
│   ├── 03_canny.ipynb
│   ├── 04_template_matching.ipynb
│   ├── 05_orb.ipynb
│   └── 06_comparison.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── annotations/
├── outputs/
│   ├── masks/
│   ├── visualisations/
│   └── metrics/
└── tests/
```

The dataset and generated outputs should not be committed to GitHub.

# Team Setup Guide

## 1. Clone the Repository

```bash
git clone https://github.com/<username>/pcb-defects-inspection.git
cd pcb-defects-inspection
```

Replace `<username>` with the GitHub username of the repository owner.

Open the project in VS Code:

```bash
code .
```

## 2. Create a Virtual Environment

To avoid Windows path-length problems, create the virtual environment outside the repository.

```powershell
python -m venv C:\venvs\pcb-env
```

Activate it:

```powershell
C:\venvs\pcb-env\Scripts\Activate.ps1
```

The terminal should display:

```text
(pcb-env)
```

If PowerShell blocks the activation command, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

## 3. Upgrade Installation Tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

## 4. Install Project Dependencies

If `requirements.txt` is available:

```powershell
python -m pip install -r requirements.txt
```

Otherwise install the required packages manually:

```powershell
python -m pip install numpy pandas matplotlib opencv-python scikit-image scikit-learn pillow pyyaml
python -m pip install notebook ipykernel streamlit pytest
```

Verify the installation:

```powershell
python -m pip check
```

## 5. Register the Jupyter Kernel

```powershell
C:\venvs\pcb-env\Scripts\python.exe -m ipykernel install --user --name pcb-env --display-name "Python (PCB Defect)"
```

Check that the kernel is registered:

```powershell
C:\venvs\pcb-env\Scripts\python.exe -m jupyter kernelspec list
```

The output should include:

```text
pcb-env
```

## 6. Select the Python Interpreter in VS Code

In VS Code:

1. Press `Ctrl + Shift + P`.
2. Select `Python: Select Interpreter`.
3. Choose:

```text
C:\venvs\pcb-env\Scripts\python.exe
```

For a notebook:

1. Open the `.ipynb` file.
2. Click `Select Kernel`.
3. Choose `Jupyter Kernel`.
4. Select `Python (PCB Defect)`.

To verify the selected interpreter, run:

```python
import sys
print(sys.executable)
```

Expected output:

```text
C:\venvs\pcb-env\Scripts\python.exe
```

## 7. Test the Environment

Create or open `notebooks/01_dataset_exploration.ipynb` and run:

```python
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import skimage
import streamlit

print("Environment setup successful")
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
```

If no error appears, the environment is ready.

## 8. Dataset Setup

The project is intended to use a paired bare-PCB image dataset such as DeepPCB.

Each sample should contain:

- a defect-free reference image;
- a defective test image;
- ground-truth annotations;
- a defect category.

Place local dataset files under:

```text
data/raw/
```

Do not commit the full dataset to GitHub.

Before implementation, verify image pairing, image resolution, annotation format, defect class names, number of samples per class, and whether the images are binary, grayscale, or colour.

## 9. Notebook Workflow

Each member should use a separate notebook for their assigned technique:

```text
01_dataset_exploration.ipynb
02_otsu.ipynb
03_canny.ipynb
04_template_matching.ipynb
05_orb.ipynb
06_comparison.ipynb
```

Each algorithm notebook should follow the same structure:

```text
Import libraries
→ Load reference and test images
→ Apply shared preprocessing
→ Run the selected technique
→ Generate mask, edge map, similarity map, or feature matches
→ Produce defect prediction
→ Measure processing time
→ Save results
→ Calculate evaluation metrics
```

## 10. Shared Output Format

All techniques should return results in a consistent format:

```python
{
    "image_id": "0001",
    "algorithm": "Otsu's Thresholding",
    "is_defective": True,
    "score": 0.87,
    "mask": defect_mask,
    "bounding_boxes": [
        {
            "x": 120,
            "y": 85,
            "width": 30,
            "height": 22
        }
    ],
    "processing_time_ms": 18.4
}
```

## 11. Evaluation Metrics

Use the same metrics for all four techniques.

### Detection Metrics

- Accuracy
- Precision
- Recall
- F1-score
- False-positive rate
- False-negative rate

### Localisation Metrics

- Intersection over Union
- Jaccard similarity
- Dice coefficient
- Boundary displacement error, where suitable

### Computational Metrics

- Processing time per image
- Mean processing time
- Standard deviation
- Frames per second

### Robustness Tests

- Gaussian noise
- Salt-and-pepper noise
- Brightness variation
- Uneven illumination
- Translation
- Rotation
- Scale variation

## 12. Streamlit Dashboard

The final dashboard will allow the user to upload a defect-free reference PCB image, upload a test PCB image, select an image processing technique, run the selected technique, and display the prediction, defect mask or matched features, detected regions, processing time, and algorithm score.

Run the dashboard from the project root:

```powershell
streamlit run app.py
```

The default local address is usually:

```text
http://localhost:8501
```

The dashboard should import reusable functions from the `algorithms/` folder rather than executing notebook cells directly.

# Git Workflow

## Create a Feature Branch

Each member should work on a separate branch:

```powershell
git checkout -b feature/otsu
```

Suggested branches:

```text
feature/otsu
feature/canny
feature/template-matching
feature/orb
feature/evaluation
feature/dashboard
```

Push the branch:

```powershell
git push -u origin feature/otsu
```

## Commit Changes

```powershell
git add .
git commit -m "Implement Otsu thresholding technique"
git push
```

Use meaningful commit messages such as:

```text
Add dataset exploration notebook
Implement Otsu thresholding
Add Canny edge comparison
Implement template matching
Add ORB feature matching
Add evaluation metrics
Create Streamlit dashboard
Fix incorrect image dimensions
```

Avoid vague messages such as `update`, `fix`, `done`, or `final`.

## Pull the Latest Main Branch

Before starting new work:

```powershell
git checkout main
git pull origin main
git checkout <your-feature-branch>
git merge main
```

# Current Development Plan

- [x] Create GitHub repository
- [x] Add `.gitignore`
- [x] Create project structure
- [x] Create Python virtual environment
- [x] Install required libraries
- [x] Register Jupyter kernel
- [ ] Add and inspect the PCB dataset
- [ ] Create dataset exploration notebook
- [ ] Implement shared image-loading functions
- [ ] Implement shared preprocessing
- [ ] Implement Otsu's thresholding
- [ ] Implement Canny edge detection
- [ ] Implement template matching
- [ ] Implement ORB feature matching
- [ ] Implement common evaluation metrics
- [ ] Run the four-technique comparison
- [ ] Conduct robustness testing
- [ ] Develop an enhanced or hybrid method
- [ ] Integrate the techniques into Streamlit
- [ ] Generate final tables, figures, and discussion

# Notes

- Do not commit the virtual environment.
- Do not commit the full PCB dataset.
- Do not commit generated masks or large output folders.
- Use the same dataset split and evaluation rules for all techniques.
- Record parameter settings for every experiment.
- Record hardware and software versions for reproducibility.
- Keep reusable code in the `algorithms/` folder.
- Use notebooks for experimentation, visualisation, and analysis.
- Use `app.py` for live dashboard execution.

## Licence

This project is developed for academic and research purposes.
