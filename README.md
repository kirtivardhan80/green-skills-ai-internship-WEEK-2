# green-skills-ai-internship-WEEK-1

# 🌳 Tree Species Identification Project

<div align="center">
  <img src="sample tree.jpg" height="2048",width="1365" alt="Tree"/>
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

---

> 🔧 Next steps: Model building, training, evaluation, and deployment.
