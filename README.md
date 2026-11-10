# Zebrafish Image Classifier (SVM + CNN)

## Description

This project is a simple image classification pipeline for thrombosis research, built using scikit-learn and PyTorch. It uses a Support Vector Machine (SVM) model and Convolutional Neural Network (CNN) model. 

The SVM model resizes all images to 15×15 pixels and flattens them into feature vectors. The dataset is then split into training and testing sets, after which an SVM classifier is trained. Model performance is evaluated using accuracy, sensitivity, specificity, balanced accuracy, and ROC AUC.

The CNN model resizes all images to 224x224 pixel image and applies a green fluorescence filter which then trains a CNN after being split into training and testing datasets. This model is then tested on accuracy, sensitivity, specificity, balanced accuracy, and ROC AUC.

**Purpose:** The approach is intended to examine clotting pattern differences between different zebrafish thrombosis models at 5 days-post-fertilization (dpf). It distinguishes between an acquired (estrogen-induced) model exhibiting speckled patterns of thrombus distribution, versus a genetic (spontaneous; protein C deficient) model exhibiting sprouting patterns with denser fluorescence signals.

## How to setup and run

### 1. Clone the repository

Clone the repo into your local directory, then navigate into it.

```bash
git clone https://github.com/ymnmurat/Zebrafish-Imaging-Classification.git
cd Zebrafish-Imaging-Classification
```

### 2. (Optional) Create a virtual environment

This is recommended to use so that dependencies in svm.py do not conflict with your local system packages.

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

Activate on macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

Run `pip install -r requirements.txt` to install dependencies.

### 4. Prepare dataset

Place your images inside the images/ folder, each corresponding to its respective subfolder. For example:

images/<br>
├── class_0/<br>
│ ├── img1.png<br>
│ ├── img2.png<br>
├── class_1/<br>
│ ├── img3.png<br>
│ ├── img4.png

**Note:** The images included in this repository are a small subset of data. The full dataset is available upon request.

The labeling of the subfolders in `/images` corresponds to the 2 groups of experimental images used for the SVM and CNN models, and are included as subfolders in the full dataset:<br>
class_0: '20220626_ProC-MePS-12'<br>
class_1: '20220704_FGBA3-MCE-MEPS-VALID-A-2-6'

### 5. Run model

To run the pipeline, use the following command:

#### SVM Model
```bash
python svm.py
```

#### CNN Model
```bash
python cnn.py
```

## Native TIFF CNN (experiment holdout)

The revised pipeline is in `cnn_native_tiff.py`. The original CNN and SVM commands above reproduce the earlier image-level split and demo. The native-TIFF analysis uses the full fluorescence TIFFs, retains dim images, and holds out entire experiments. The default full-frame model is the primary Figure 2B model. Use `--view crop` to evaluate the same CNN on a 960 × 384 µm tail window estimated from local texture.

Install the analysis dependencies.

```bash
python -m pip install numpy Pillow scipy scikit-learn torch
```

Use an ordered JSON array of image records from the study data. Each record needs `member` (a TIFF path relative to `--root`, or an entry name in `archive`), `experiment`, `split` (`train`, `historical_test`, or `recent`), `reference_class` (0 for ProC, 1 for FGBA3, or `null` for recent), and `um_per_pixel` (native X pixel size). For ZIP inputs set `archive` to the ZIP path relative to `--root`. Set `mestranol_uM` on recent ProC rows. Mark excluded rows with `split: "excluded"` or leave them out of the manifest. Records marked `M20_challenge` or `duplicate_of` are skipped. The script checks that retained experiments and image content do not overlap across splits.

```bash
python cnn_native_tiff.py --manifest study_manifest.json --root /path/to/study-data --output run_full --view full
```

The Figure 2B study manifest contains 1,276 training and 781 historical test images. The 665 recent ProC images are scored separately. A reference label is supplied only for the untreated recent controls. The script writes a PyTorch checkpoint, per-image predictions, historical metrics, and the prepared image cache to the specified new output directory. The ROC AUC uses continuous Class 1 scores, and the binary-call AUC is reported separately. For the study acquisitions, a native pitch below 0.9 µm/pixel identifies the 20× subset. The pilot used CPU PyTorch 2.6.0, NumPy 2.2.2, Pillow 11.1.0, SciPy 1.15.1, and scikit-learn 1.6.1. Use the full study data and original manifest order to reproduce the reported numbers.
