# PCB Defect Inspection Using Classical Image Processing

## Project Overview

This project develops and compares four classical image processing techniques for bare printed circuit board (PCB) defect inspection:

1. Otsu's Thresholding
2. Canny Edge Detection
3. Template Matching
4. ORB Feature Matching

The system compares a defect-free PCB template with a defective PCB image to identify structural differences. All four techniques will be evaluated under the same experimental conditions using accuracy, precision, recall, F1-score, localisation performance, and processing time.

The six defect categories are:

- Missing hole
- Mouse bite
- Open circuit
- Short circuit
- Spur
- Spurious copper

The final stage may combine the strongest techniques into an enhanced or hybrid method and integrate the system into a Streamlit dashboard.

## Dataset

Dataset source:

```text
https://www.kaggle.com/datasets/akhatova/pcb-defects/data
```

The dataset contains normally aligned images, rotated images, XML annotations, and ten defect-free PCB templates. For the first comparison, use only the normally aligned images. Use the rotated subset later for robustness testing.

Store the dataset at:

- **macOS / Linux**:
  ```text
  ~/Documents/Project/pcb-defects-inspection/data/raw/PCB_DATASET
  ```
- **Windows**:
  ```text
  C:\...\pcb-defects-inspection\data\raw\PCB_DATASET
  ```

Expected structure:

```text
PCB_DATASET/
├── Annotations/
│   ├── Missing_hole/
│   ├── Mouse_bite/
│   ├── Open_circuit/
│   ├── Short/
│   ├── Spur/
│   └── Spurious_copper/
├── Images/
│   ├── Missing_hole/
│   ├── Mouse_bite/
│   ├── Open_circuit/
│   ├── Short/
│   ├── Spur/
│   └── Spurious_copper/
├── PCB_USED/
└── rotation/
```

Do not upload the dataset to GitHub.

## Project Structure

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
│   ├── 01_dataset_exploration_and_preprocessing.ipynb
│   ├── 02_otsu.ipynb
│   ├── 03_canny.ipynb
│   ├── 04_template_matching.ipynb
│   ├── 05_orb.ipynb
│   └── 06_comparison.ipynb
├── data/
│   ├── raw/
│   │   └── PCB_DATASET/
│   └── processed/
├── outputs/
│   ├── masks/
│   ├── visualisations/
│   └── metrics/
└── tests/
```

# Team Setup

## 1. Clone the Repository

```bash
git clone https://github.com/<username>/pcb-defects-inspection.git
cd pcb-defects-inspection
code .
```

## 2. Create and Activate the Virtual Environment

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows (PowerShell)

```powershell
python -m venv C:\venvs\pcb-env
C:\venvs\pcb-env\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
C:\venvs\pcb-env\Scripts\Activate.ps1
```

## 3. Install Dependencies

### macOS / Linux

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
```

If `requirements.txt` is unavailable:

```bash
python3 -m pip install numpy pandas matplotlib opencv-python scikit-image scikit-learn pillow pyyaml
python3 -m pip install notebook ipykernel streamlit pytest
```

Verify:

```bash
python3 -m pip check
```

### Windows (PowerShell)

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If `requirements.txt` is unavailable:

```powershell
python -m pip install numpy pandas matplotlib opencv-python scikit-image scikit-learn pillow pyyaml
python -m pip install notebook ipykernel streamlit pytest
```

Verify:

```powershell
python -m pip check
```

## 4. Register the Jupyter Kernel

### macOS / Linux

```bash
python3 -m ipykernel install --user --name pcb-env --display-name "Python (PCB Defect)"
```

Check registration:

```bash
python3 -m jupyter kernelspec list
```

### Windows (PowerShell)

```powershell
C:\venvs\pcb-env\Scripts\python.exe -m ipykernel install --user --name pcb-env --display-name "Python (PCB Defect)"
```

Check registration:

```powershell
C:\venvs\pcb-env\Scripts\python.exe -m jupyter kernelspec list
```

## 5. Select the Interpreter in VS Code

### macOS / Linux

Select:

```text
./venv/bin/python
```

### Windows

Select:

```text
C:\venvs\pcb-env\Scripts\python.exe
```

For notebooks on both platforms, select the kernel:

```text
Python (PCB Defect)
```

# Dataset Exploration and Shared Preprocessing

Use:

```text
notebooks/01_dataset_exploration_and_preprocessing.ipynb
```

# Evaluation Metrics

Detection metrics:

- Accuracy
- Precision
- Recall
- F1-score
- False-positive rate
- False-negative rate

Localisation metrics:

- Intersection over Union
- Dice coefficient
- Bounding-box overlap
- Detection rate per defect instance

Computational metrics:

- Processing time per image
- Mean processing time
- Standard deviation
- Frames per second

Robustness tests:

- Rotation
- Translation
- Gaussian noise
- Salt-and-pepper noise
- Brightness variation
- Uneven illumination
- Scale variation

# Streamlit Dashboard

The final dashboard may allow users to upload a reference image and test image, select an algorithm, inspect predicted masks and boxes, and view processing time and evaluation results.

Run it with:

```bash
streamlit run app.py
```

# Git Workflow

Create a feature branch:

```bash
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

Commit changes:

```bash
git add .
git commit -m "Add dataset exploration and shared preprocessing"
git push
```

# Important Notes

- Do not commit the dataset.
- Do not commit the virtual environment.
- Do not commit large generated outputs.
- Use the same shared preprocessing for all four algorithms.
- Keep algorithm-specific enhancements out of the baseline preprocessing.
- Keep experiments and visualisations in notebooks.
- Move reusable functions into `algorithms/`.
- Record all parameters and processing times.
- Verify that each defective image maps to the correct PCB template.
- Use rotated images only after the aligned baseline works.

## Licence

This project is developed for academic and research purposes.
