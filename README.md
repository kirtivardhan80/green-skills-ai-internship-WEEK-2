# green-skills-ai-internship-AICTE

# 🌳 Tree Species Identification Project

<div align="center">
  <img src="sample tree.jpg" height="2048" width="1365" alt="Tree"/>
  <br/>
</div>

---

## 🔍 Project Overview

The **Tree Species Identification Project** leverages modern **AI and machine learning techniques** to automate the classification of tree species based on leaf images. Traditionally, this required expert botanical knowledge — this deep learning-based solution makes the process faster, scalable, and more accessible for ecological monitoring and smart agriculture.

---

## 🎯 Objectives

- 📸 Classify tree species using leaf images.
- ⚙️ Automate identification using computer vision and deep learning.
- 🌾 Support ecological research and smart farming initiatives.

---

## 🗂️ Dataset

- **Source**: [Kaggle – Tree Species Identification Dataset](https://www.kaggle.com/datasets/viditgandhi/tree-species-identification-dataset)
- Thousands of leaf images categorized into 31 species-specific folders.
- Each folder represents a different tree species.

---

## ✅ Work Completed

### 1️⃣ Dataset Loading
- Dataset was downloaded and loaded using Keras `ImageDataGenerator` with 80-20 split.

### 2️⃣ Data Visualization
- Sample images and class distribution were visualized using matplotlib.

### 3️⃣ Data Preprocessing
- Corrupted images were detected and removed using PIL.
- System folders like `.git` were safely skipped.

### 4️⃣ Model Building & Training
- MobileNetV2 was used as the base model with transfer learning.
- Model was trained for 25 epochs and validated successfully.

### 5️⃣ Fine-Tuning
- Upper layers of the base model were unfrozen and fine-tuned for additional performance.
- ✅ Final model was saved after fine-tuning for deployment.

### 6️⃣ Testing on Custom Images
- The model was tested on new user-uploaded images to validate real-world performance.

### 7️⃣ Streamlit Web App
- A simple and interactive web interface was created using Streamlit to upload and predict leaf species.

---

## 💡 Improvisations Done By Me

- Built a complete preprocessing pipeline to clean the dataset.
- Used transfer learning and fine-tuning on MobileNetV2 for optimized training.
- Designed and implemented a working Streamlit app for predictions.
- Ensured the model was saved after final training for deployment.

---

## 📦 Files Included

| File/Folder                    | Description |
|-------------------------------|-------------|
| `tree_species_model_final.h5` | Final trained model (MobileNetV2) |
| `app.py`                      | Streamlit web app to predict species |
| `requirements.txt`            | Python packages needed to run app |
| `tree_dataset/`               | Local dataset (not uploaded on GitHub) |
| `sample tree.jpg`             | Sample image used in README banner |

---

## 📥 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/tree-species-identification.git
cd tree-species-identification
```

### 2. Create a virtual environment (recommended)

```bash
conda create -n treespecies python=3.9
conda activate treespecies
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🧠 Model Details

- Base Model: **MobileNetV2** (ImageNet pretrained)
- Classifier: Custom top layers (Dense + Dropout)
- Loss: Categorical Crossentropy
- Optimizer: Adam
- Accuracy Achieved: ~70% (validation)

---

## 🔮 Future Work

- Add confidence score display for predictions.
- Deploy the Streamlit app on [Streamlit Cloud](https://share.streamlit.io/)
- Explore Grad-CAM or LIME for explainability.
- Expand dataset with real-world images for testing.

---

## 🙌 Acknowledgements

- Dataset by [Vidit Gandhi on Kaggle]([https://www.kaggle.com/datasets/viditgandhi/tree-species-identification-dataset](https://www.kaggle.com/datasets/viditgandhi/tree-species-identification-dataset))
- Internship organized by **Edunet Foundation**, supported by **AICTE** and **Shell India**

---

> Made with ❤️ by Kirti Vardhan Singh during the Green Skills AI Internship (AICTE–Shell)
