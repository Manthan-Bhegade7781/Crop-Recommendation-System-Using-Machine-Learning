# Crop-Recommendation-System-Using-Machine-Learning
This project predicts the most suitable crop for cultivation based on soil nutrients (N, P, K), temperature, humidity, pH, and rainfall using various machine learning models. The best-performing model is deployed and saved for future prediction 

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

Seven machine learning algorithms were tested:

- 🌳 **Random Forest** (**99.55%**)
- 🌲 Decision Tree
- ⚙️ XGBoost
- 📐 Support Vector Machine (SVM)
- 🧮 K-Nearest Neighbors (KNN)
- 🧠 Gaussian Naive Bayes
- 📈 Logistic Regression

📌 The **Random Forest** algorithm performed best, offering excellent **Precision, Recall, and F1-score**.

---

## 📐 Model Architecture

The model uses **Scikit-learn** and **Pandas** for data handling, with preprocessing steps like:

- Normalization
- Label Encoding
- Train-Test Splitting

Once trained, it predicts the best crop for given soil and climate data.

**Tech Stack**:
- 🐍 Python 3.8+
- 🧪 Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, XGBoost
- 🌐 Interface: **Streamlit** web app

---

## 🖼 System Architecture

```text
[User Inputs Data] → [Preprocessing & ML Model] → [Best Crop Output Shown on Interface]



