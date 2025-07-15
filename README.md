# 🧠 3D Skin Cancer Detection using Synthetic Depth Augmentation and Deep Learning

## 📌 Overview

This project aims to detect skin cancer using deep learning models applied on both 2D dermoscopic images and synthetically generated 3D volumetric representations. By leveraging state-of-the-art CNN architectures and synthetic 3D augmentation from single 2D images, we improve classification accuracy and provide a novel approach to cancer detection.

* **2D Ensemble Model Accuracy:** 92%
* **3D CNN Model Accuracy:** 84%

The combination of both planar and depth analysis allows for robust predictions and enhances diagnostic reliability.

---

## 📁 Project Structure

```
skin-cancer-3d-hybrid/
│
├── Skin Cancer detection.ipynb     
├── README.md                         
├── requirements.txt                  
├── Research Paper.pdf                                                

```

---

## 📦 Dataset

**Source:** [ISIC 2024 Challenge Dataset](https://www.kaggle.com/competitions/isic-2024-challenge)

* **Images:** High-resolution dermoscopic images
* **Metadata Fields:** lesion ID, diagnosis hierarchy, patient demographics, 3D TBP coordinates, and image characteristics
* **Classes:**

  * `0` → Benign
  * `1` → Malignant

---

## 🔍 Problem Statement

Detect melanoma and other forms of skin cancer using a hybrid approach:

1. 2D ensemble modeling on original dermoscopic images.
2. 3D CNN modeling using depth-augmented volumes from single 2D reference images.

This project helps improve classification performance and clinical interpretability in non-clinical or telehealth settings.

---

## 🧠 Model Architecture

### 1. **2D Ensemble Model**

* **Base Models**: EfficientNetV2B0, ResNet50
* **Architecture**:

  * Feature extraction from both networks
  * Concatenated features
  * Fully connected dense layers
* **Output**: Binary classification using sigmoid activation

### 2. **3D CNN Model**

* Input: 3D volume generated via augmented depth slices
* Layers:

  * Conv3D layers with BatchNorm and MaxPooling
  * Fully connected dense layers
* Output: Binary prediction (Benign or Malignant)

---

## 🧪 Training Strategy

* **Data Sampling**: Class imbalance handled using targeted oversampling for malignant class with data augmentation
* **Train/Validation Split**: 80/20 stratified split
* **Loss Function**: Binary Crossentropy
* **Optimizer**: Adam
* **Metrics**: Accuracy and AUC

---

## 🧱 3D Augmentation Pipeline

* Used **ImageDataGenerator** to create 16 depth slices per image
* Slices were stacked into `(256, 256, 16, 1)` volumetric tensors
* Rendered into full 3D mesh using Marching Cubes
* Used as input to 3D CNN

---

## 🎯 Inference Pipeline

1. **Preprocess test image**
2. **Make 2D prediction using ensemble model**
3. **Generate 3D volume from 2D image**
4. **Make 3D prediction**
5. **Output both predictions**

---

## 📈 Results

| Model       | Accuracy | AUC  |
| ----------- | -------- | ---- |
| 2D Ensemble | 92%      | 0.96 |
| 3D CNN      | 84%      | 0.89 |

---

## 🛠 Tools & Frameworks

* **TensorFlow/Keras**
* **OpenCV**
* **NumPy, Pandas**
* **Matplotlib, Seaborn**
* **scikit-learn**
* **scikit-image**
* **Google Colab / Kaggle Notebooks**

---

## 🚀 How to Run

1. **Clone the repository**

```bash
   git clone https://github.com/yourusername/skin-cancer-3d-hybrid.git
   cd skin-cancer-3d-hybrid
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Download the dataset** from [Kaggle](https://www.kaggle.com/competitions/isic-2024-challenge) and place it in `/dataset/`.

4. **Run the notebook**

```bash
jupyter notebook notebooks/skin_cancer_detection.ipynb
```

---

## 👨‍💻 Author

**Aaryan Sinha**
Final Year Computer Science Engineering Student
Research Intern at IIT Indore (Deep Learning - Image Processing)
