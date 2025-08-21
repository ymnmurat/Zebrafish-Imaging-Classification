import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, accuracy_score, balanced_accuracy_score, roc_curve, auc

data_dir = 'images'

# Read images from data folders & convert into matrix, then store it into the list
categories = ['class_0', 'class_1']
data = [] # Vector form
labels = [] # 0 or 1
for category_idx, category in enumerate(categories):
  for file in os.listdir(os.path.join(data_dir, category)):
    img_path = os.path.join(data_dir, category, file)
    #print(img_path)
    img = imread(img_path)
    img = resize(img, (15,15))
    data.append(img.flatten())
    labels.append(category_idx)

# Convert list of matrixes into numpy array:
data = np.asarray(data)
labels = np.asarray(labels)

# Split data into training & testing
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Build svm classifier
svm_model = SVC()
svm_model.fit(x_train, y_train)

y_pred = svm_model.predict(x_test)
print(y_pred)

# Evaluate the model
score = accuracy_score(y_test, y_pred)
print("Accuracy score:", score)

# Calculate sensitivity, specificity, and balanced accuracy
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
balanced_accuracy = (sensitivity + specificity) / 2

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Print results
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("Balanced Accuracy:", balanced_accuracy)
print("ROC AUC:", roc_auc)