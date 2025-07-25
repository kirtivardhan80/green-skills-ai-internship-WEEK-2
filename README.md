# green-skills-ai-internship-AICTE

# 🌳 Tree Species Identification Project

<div align="center">
  <img src="sample tree.jpg" height="2048" width="1365" alt="Tree"/>
  <br/>
</div>

## 🔍 Project Overview
The Tree Species Identification Project aims to leverage modern **AI and machine learning techniques** to automate the classification of tree species based on leaf images. Traditionally, tree identification required expert botanical knowledge, but this project uses a **deep learning-based approach** to make the process faster, more accurate, and scalable.

---

## 🎯 Objectives
- Develop a system to classify tree species using leaf images.
- Automate the process using computer vision and deep learning.
- Provide a foundation for ecological monitoring and smart agriculture applications.

---

## 🗂️ Dataset
- **Source**: [Kaggle – Tree Species Identification Dataset](https://www.kaggle.com/datasets/viditgandhi/tree-species-identification-dataset)
- The dataset contains thousands of images of tree leaves, organized in subfolders by species.
- Each folder represents a different species or disease class.

---

## ✅ Work Completed

### 1. Dataset Loading
- The dataset was downloaded locally and extracted.
- Images were loaded using Keras’s `ImageDataGenerator`.
- An **80-20 split** was created for training and validation using the `validation_split` parameter.
- All images were resized to **224×224** for model compatibility.

### 2. Data Visualization
- Displayed **random image samples** with class labels to verify dataset quality.
- Plotted a **bar chart of class distribution** to assess balance among categories.
- Verified that the dataset is properly structured and ready for model training.

### 3. Data Preprocessing
- Identified and safely removed **corrupted images** using PIL.
- Handled `.git` system directory issues and ensured only image files were processed.
- (Optional) Verified that no duplicate or overly small/large images affected training.

### 4. Model Building and Training
- Used **MobileNetV2** with transfer learning for efficient training.
- Trained a custom classifier on top of the frozen base model.
- Achieved **~70% validation accuracy** after 25 epochs.

### 5. Fine-Tuning
- Unfroze top layers of MobileNetV2.
- Recompiled the model with a lower learning rate.
- Fine-tuned the model to improve accuracy and adaptability.

---

### 💡 Improvisations Done By Me
I implemented a full preprocessing pipeline including detection and removal of corrupted images, ensured system files like `.git` were excluded, and added data augmentation for better generalization. I applied **transfer learning** with MobileNetV2 and fine-tuned the model to significantly improve validation accuracy, making the model more robust to real-world data.

---

> 🚀 Next steps: Test on new images, evaluate with confusion matrix, and deploy the model using Streamlit or Flask.
