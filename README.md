# 🌾 Crop Recommendation System: Your Smart Agriculture Companion 

---

## 📌 Overview

The **Crop Recommendation System** is a smart farming tool that leverages machine learning to help farmers choose the best crops to cultivate based on soil and environmental conditions. Using key parameters like **Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH**, and **Rainfall**, the system provides precise crop suggestions.

With a clean interface and real-time analysis, this system improves decision-making, increases crop yield, and promotes sustainable farming practices. The model is built using advanced algorithms, with **Random Forest achieving 99.55% accuracy**.

---

## 📊 Features

✅ **Smart Crop Recommendation**  
• Recommends the most suitable crop using 7 key environmental parameters and ML classification models.

✅ **User-Friendly Interface**  
• Easy-to-use web app that allows users to input data and get instant recommendations.

✅ **High Accuracy**  
• Random Forest algorithm delivers a prediction accuracy of **99.55%**.

---

## 🧪 Dataset

The dataset contains **2,200 entries** and **8 columns**:

| Feature       | Description                         |
|---------------|-------------------------------------|
| `N`           | Nitrogen content in soil            |
| `P`           | Phosphorus content in soil          |
| `K`           | Potassium content in soil           |
| `Temperature` | Temperature in degrees Celsius      |
| `Humidity`    | Relative humidity in percentage     |
| `pH`          | pH value of the soil                |
| `Rainfall`    | Rainfall in mm                      |
| `Label`       | Crop type (Target variable)         |

This labeled data helps the machine learning model learn patterns to recommend crops based on environmental factors.

---

## 📦 Crop Recommendation Model

Six machine learning algorithms were tested:

- 🌳 **Random Forest** (**99.55%**)
- 🌲 Decision Tree
- ⚙️ XGBoost
- 📐 Support Vector Machine (SVM)
- 🧮 K-Nearest Neighbors (KNN)
- 📈 Logistic Regression

📌 The **Random Forest** algorithm performed best, offering excellent **Precision, Recall, and F1-score**.

![Comparison bar](https://github.com/user-attachments/assets/94865916-49ba-4ba7-bf45-a3b79efcb322)

---

## 📐 Model Architecture

The model uses **Scikit-learn** and **Pandas** for data handling, with preprocessing steps like:

- Normalization
- Label Encoding
- Train-Test Splitting

  
![work flow](https://github.com/user-attachments/assets/42f99322-9283-41c9-86e7-ff82d02549ed)

Once trained, it predicts the best crop for given soil and climate data.

**Tech Stack**:
- 🐍 Python 3.8+
- 🧪 Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, XGBoost
- 🌐 Interface: **Streamlit** web app

---

## 🧩 Integration

The crop model is integrated into the **AgriSens** web app, providing real-time crop suggestions. The system is designed to be extended with additional features like:

- 🌿 Plant Disease Detection
- 🌱 Fertilizer Recommendation

---

## 🎯 Outcomes

- ✅ Accurate and reliable crop suggestions
- 🌐 Easy accessibility through web interface
- 🌱 Supports sustainable and profitable farming

---

## 📌 Copyright

- Copyright Form Diary No: 8848/2025-CO/L

---
## 📬 Contact

- manthanbhegade407@gmail.com
  
---
