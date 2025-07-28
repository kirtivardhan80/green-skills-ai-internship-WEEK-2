# green-skills-ai-internship-AICTE

# 🌳 Tree Species Identification Project

<div align="center">
  <img src="sample tree.jpg" height="2048" width="1365" alt="Tree"/>
  <br/>
</div>

---

## 🔍 Project Overview

The **Tree Species Identification Project** aims to leverage modern **AI and machine learning techniques** to automate the classification of tree species based on leaf images. Traditionally, this required expert botanical knowledge — this deep learning-based solution makes the process faster, scalable, and more accessible for ecological monitoring and smart agriculture.

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
- Downloaded and extracted locally.
- Loaded using Keras `ImageDataGenerator` with an **80-20 train-validation split**.
- All images resized to **224×224** for model compatibility.

### 2️⃣ Data Visualization
- Displayed random image samples with their class labels.
- Plotted class distribution using `matplotlib`.
- Verified proper structure and labeling of dataset.

### 3️⃣ Data Preprocessing
- Detected and removed **corrupted images** using PIL.
- Excluded system folders like `.git`.
- Ensured only valid image files were processed.

### 4️⃣ Model Building, Training 
- Used **MobileNetV2** as base model with transfer learning.
- Built custom top layers (GlobalAveragePooling, Dense, Dropout).
- Trained with base model frozen for 25 epochs (achieved ~70% validation accuracy).
- Fine-tuned top layers with a lower learning rate for better generalization.

---

### 5️⃣ Fine-Tuning & Saving
- Unfroze upper layers of the MobileNetV2 base.
- Recompiled with a smaller learning rate (`1e-5`).
- Trained for 10 additional epochs.
- Improved performance and adaptability to dataset.
- ✅ **Saved final model** using:

```python
model.save("tree_species_model_final.h5")
```

### 6️⃣ Testing on Custom Images
- Allowed users to upload new leaf images for prediction.
- Resized, normalized, and preprocessed images (shape: 224×224×3).
- Mapped predictions to correct class labels.
- ✅ Displayed prediction with **confidence score**:

```python
pred = model.predict(img_array)
predicted_class = np.argmax(pred[0])
confidence = float(np.max(pred[0])) * 100
```

---

### 7️⃣ Streamlit Web App Deployment
- Built a Streamlit app (`app.py`) to serve predictions through a web UI.
- Users can upload any JPG/PNG leaf image and get instant classification.
- The model is loaded in `.h5` format and predictions are made on-the-fly.
- Displays both species and model confidence percentage.

```bash
streamlit run app.py
```

---

## 💡 Improvisations Done By Me

- Implemented a full **image preprocessing pipeline** to detect/remove corrupted files.
- Used `os` and `PIL` to skip non-image folders (like `.git`) and invalid files.
- Applied **transfer learning + fine-tuning** on MobileNetV2 for robust training.
- Built and deployed a **Streamlit web app** with confidence-based prediction output.

---

## 📦 Files Included

| File/Folder                    | Description |
|-------------------------------|-------------|
| `tree_species_model_final.h5` | Final trained model (MobileNetV2) |
| `app.py`                      | Streamlit web app to predict species |
| `requirements.txt`            | Python packages needed to run app |
| `tree_dataset/`               | Local dataset with 31 species (not included in repo) |
| `sample tree.jpg`             | Sample image used in README banner |

---

## 📥 Installation & Setup

```bash
git clone https://github.com/yourusername/tree-species-identification.git
cd tree-species-identification
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 Model Details

- Architecture: **MobileNetV2** (pretrained on ImageNet)
- Loss Function: `categorical_crossentropy`
- Optimizer: `Adam`
- Accuracy Achieved: ~70% (validation)
- Confidence scores provided in prediction results

---

## 🔮 Future Work

- 🎯 Deploy to [Streamlit Cloud](https://share.streamlit.io/)
- 📱 Build a mobile-compatible UI
- 📊 Add model explainability (Grad-CAM, LIME)
- 🧪 Evaluate on a real-world image dataset (not just Kaggle)

---

## 🙌 Acknowledgements

- Dataset by [Vidit Gandhi on Kaggle](https://www.kaggle.com/datasets/viditgandhi/tree-species-identification-dataset)
- Internship and guidance by **Edunet Foundation**, **AICTE**, and **Shell India**

---

> Made with 🌿 by Kirti Vardhan Singh during the Green Skills AI Internship (AICTE–Shell)
