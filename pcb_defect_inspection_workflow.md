# PCB Defect Inspection Project Workflow

## 1. Project scope

The project compares four classical computer-vision techniques for PCB defect inspection: Otsu thresholding, Canny edge detection, template matching, and ORB feature matching. All four methods must be evaluated in their **raw form**. Do not add morphology, contour merging, filtering, denoising beyond the shared basic preprocessing, or any other post-processing intended to improve results. This keeps the comparison transparent and fair.

## 2. Dataset split

Use the **693 normally aligned images** for the main experiment and keep rotated images separate for an optional robustness test. Divide the aligned images as follows:

| Portion | Percentage | Image count | Purpose |
|---|---:|---:|---|
| Development | 20% | 139 | Implement and debug the algorithms |
| Validation | 20% | 139 | Select algorithm parameters and decision thresholds |
| Test | 60% | 415 | Perform the final unbiased comparison |
| **Total** | **100%** | **693** | |

The split should be stratified so that the six defect classes are represented as evenly as possible in every portion. If multiple files are derived from the same PCB sample, board, scene, or source pair, assign the whole group to one portion to prevent data leakage.

### Reproducible manifest-based splitting

Do **not** ask each member to copy or divide the files manually. Instead, one designated team member should generate a single manifest, such as `dataset_split.csv`, and commit or share that file with the team. The source images can remain in their original folders.

Recommended manifest columns:

```text
image_id,image_path,reference_path,annotation_path,defect_class,group_id,split
0001,data/images/0001.jpg,data/templates/0001.jpg,data/xml/0001.xml,missing_hole,board_01,development
```

Generate the manifest using these rules:

1. Scan and sort all 693 aligned image records by a stable identifier.
2. Verify that every record has the required reference image and annotation.
3. Group related records using `group_id` when necessary.
4. Perform a class-stratified, group-aware split with a fixed random seed, for example `seed = 42`.
5. Adjust only for unavoidable rounding so the final counts are exactly 139 development, 139 validation, and 415 test images.
6. Save the chosen split for every image in `dataset_split.csv`.
7. Record the seed, generation date, source dataset version, and manifest checksum in the project README.
8. Treat the manifest as fixed after experiments begin. Any correction should create a new clearly versioned manifest and require the whole team to rerun affected experiments.

Every notebook, evaluation script, and Streamlit prototype should load the same manifest and select rows by the `split` column. Team members may create local views, links, or temporary folders from the manifest if their tools require folders, but the manifest remains the authoritative split. The test labels and results should not be inspected while developing or tuning the methods.

## 3. Dataset exploration

Use `01_dataset_exploration_and_preprocessing.ipynb` to:

```text
Inspect the folder structure
→ count images in each defect class
→ verify that 693 aligned records are present
→ inspect representative samples
→ parse the XML annotations
→ draw and verify ground-truth bounding boxes
→ confirm defective-image, reference-image, and annotation pairing
→ verify the manifest counts and class distribution
```

The outputs should include a class-distribution table, split-distribution table, sample visualisations, and confirmation that all paths and annotations are valid.

## 4. Shared preprocessing

Store reusable loading and preprocessing functions in `algorithms/common.py`. Apply the same minimal preprocessing wherever it is applicable:

```text
Load reference image and defective image
→ verify their dimensions and alignment
→ resize only when required by the agreed input specification
→ convert to grayscale when required by the algorithm
→ produce the algorithm-specific comparison input
```

Do not use Gaussian blur, morphology, connected-component merging, contour filtering, or other enhancement/refinement stages. Any unavoidable library defaults and all parameter values must be documented.

## 5. Raw algorithm implementation

Use the development set to implement and debug the four methods. Keep a separate notebook for each method.

### 5.1 Raw Otsu thresholding — `02_otsu.ipynb`

```text
Reference and defective grayscale images
→ absolute difference image
→ Otsu automatic threshold
→ raw binary defect mask
→ direct contour extraction
→ predicted defect boxes
```

### 5.2 Raw Canny edge detection — `03_canny.ipynb`

```text
Reference and defective grayscale images
→ apply Canny independently using fixed thresholds
→ absolute difference between the two raw edge maps
→ direct contour extraction
→ predicted defect boxes
```

Do not connect broken edges, dilate edges, or merge nearby regions.

### 5.3 Raw template matching — `04_template_matching.ipynb`

```text
Reference image and defective image
→ compute the selected raw template-matching similarity response
→ apply the chosen validation-set decision threshold
→ direct extraction of low-similarity regions
→ predicted defect boxes
```

Do not smooth the response map, merge detections, or apply non-maximum suppression unless the chosen library operation intrinsically requires it; any such requirement must be disclosed consistently.

### 5.4 Raw ORB feature matching — `05_orb.ipynb`

```text
Reference image and defective image
→ detect ORB keypoints and descriptors
→ perform descriptor matching with the selected raw matcher
→ identify unmatched or inconsistent feature locations using the chosen validation rule
→ directly form predicted defect locations or boxes
```

Do not cluster, expand, merge, or morphologically refine the detected regions.

## 6. Validation and parameter selection

Run the completed raw implementations on the validation set. Select only the parameters inherently required by each method, such as Canny thresholds, template-matching score threshold, ORB feature/match settings, and the rule used to convert raw outputs to detections. Freeze every parameter after validation. Do not repeatedly check the test set or alter parameters in response to test results.

Maintain a configuration file containing the final frozen parameters so every team member evaluates exactly the same implementations.

## 7. Final test and comparison

Run each frozen algorithm once on all 415 test images. Compare predicted detections with XML ground truth using the same evaluation code and matching rule for all algorithms. Report suitable metrics such as precision, recall, F1-score, intersection over union, detection rate by defect class, false positives per image, and average processing time per image.

Save one result record per image and algorithm, including the manifest image ID, algorithm version, parameter configuration, predicted boxes, metrics, runtime, and success/error status. Include example true positives, false positives, and false negatives in the final report without using them to retune the algorithms.

## 8. Optional rotated-image robustness test

After the main aligned-image comparison is complete, run the already frozen algorithms on the separately held rotated images. Report these results as a robustness experiment, not as part of parameter selection or the main 693-image test.

## 9. Streamlit prototype

Implement the prototype only after the four algorithms and their parameters are frozen. A simple user flow is:

```text
User uploads a reference PCB image and a test PCB image
→ application verifies file type and compatible dimensions
→ user selects one of the four raw algorithms
→ application loads the frozen configuration
→ selected raw pipeline runs
→ application displays the input images, raw output mask/map, predicted defect boxes, and runtime
→ user may download a result image or summary
```

The prototype must call the same reusable algorithm functions used for evaluation rather than duplicate notebook code. Clearly label outputs as raw detections and show a helpful message when no descriptors, matches, or defect regions are found.

## 10. Recommended project structure

```text
project/
├── data/
│   └── dataset_split.csv
├── algorithms/
│   ├── common.py
│   ├── otsu.py
│   ├── canny.py
│   ├── template_matching.py
│   └── orb.py
├── configs/
│   └── frozen_parameters.yaml
├── notebooks/
│   ├── 01_dataset_exploration_and_preprocessing.ipynb
│   ├── 02_otsu.ipynb
│   ├── 03_canny.ipynb
│   ├── 04_template_matching.ipynb
│   ├── 05_orb.ipynb
│   └── 06_final_evaluation.ipynb
├── results/
├── app.py
└── README.md
```

## 11. Complete flow summary

```text
Verify the 693 aligned records
→ generate and freeze one shared 139/139/415 manifest
→ explore the dataset and validate annotations
→ implement shared minimal preprocessing
→ build four raw algorithms on the development set
→ choose inherent parameters on the validation set
→ freeze code and configurations
→ run one final evaluation on the test set
→ compare accuracy and runtime
→ optionally test robustness on rotated images
→ integrate the frozen functions into Streamlit
```
