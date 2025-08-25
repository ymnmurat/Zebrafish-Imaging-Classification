# Zebrafish Image Classifier (SVM + CNN)

## Description

This project is a simple image classification pipeline for thrombosis research, built using scikit-learn. It uses a Support Vector Machine (SVM) model and Convolutional Neural Network (CNN) model. 

The model resizes all images to 15×15 pixels and flattens them into feature vectors. The dataset is then split into training and testing sets, after which an SVM classifier is trained. Model performance is evaluated using accuracy, sensitivity, specificity, balanced accuracy, and ROC AUC.

This model resizes all images to 1000x1000 pixel image which then trains a CNN after being split into training and testing datasets. This model is then tested on accuracy, sensitivity, specificity, balanced accuracy, and ROC AUC.

**Purpose:** The model is intended to examine clotting pattern differences between different thrombosis models. It distinguishes between an acquired model exhibiting speckled patterns of thrombus distribution, versus a genetic model exhibiting sprouting patterns with denser fluorescence signals.

## How to setup and run

### 1. Clone the repository

Clone the repo into your local directory, then navigate into it.

```bash
git clone https://github.com/dwu1011/Zebrafish-Imaging-Classification.git
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

**Note:** The images included in this repository are a small subset of data. The full dataset can be downloaded here: [https://drive.google.com/drive/folders/1xrwQSTaHyiDmvbMnMK4_lE2wCr7X80gU?usp=sharing]

The labeling of the subfolders in `/images` corresponds to the 2 groups of experimental images used for this SVM model, and are included as subfolders in the full dataset:<br>
class_0: '20220626_ProC-MePS-12'<br>
class_1: '20220704_FGBA3-MCE-MEPS-VALID-A-2-6'

Other subfolders in the full dataset were not used in training and testing this model, but are included in the link for your usage.

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