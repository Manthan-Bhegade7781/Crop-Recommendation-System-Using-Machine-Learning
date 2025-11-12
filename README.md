# 🌾 Crop Recommendation System using Machine Learning

An intelligent web-based application that predicts the most suitable crop for cultivation based on soil nutrients and environmental parameters using **Machine Learning**.  
This project integrates a **React.js (Vite)** frontend, **Flask (Python)** backend, and trained ML models to provide accurate, real-time recommendations.

---

## 📘 Project Overview

The **Crop Recommendation System** is designed to assist farmers and agricultural researchers in selecting the most suitable crop based on soil composition and environmental factors.  
By analyzing inputs such as **Nitrogen (N)**, **Phosphorus (P)**, **Potassium (K)**, **temperature**, **humidity**, **pH**, and **rainfall**, the system predicts which crop will yield the best results.  

This combination of **Machine Learning and Web Technologies** supports smarter farming decisions and promotes **sustainable agriculture**.

---

## 🧠 Problem Statement

Selecting an appropriate crop based on soil and climate conditions is crucial but often depends on human intuition or outdated methods.  
This project solves that by leveraging **Machine Learning models** trained on real-world agricultural data to recommend the best crop based on given conditions.

---

## 🎯 Objectives

- To recommend the most suitable crop based on soil and environmental parameters.  
- To train multiple ML algorithms and evaluate their performance.  
- To integrate React frontend with Flask backend for real-time predictions.  
- To design a responsive, user-friendly web interface.  
- To promote data-driven smart farming practices.

---

## ⚙️ Features

✅ Predicts the best crop using ML models  
✅ Real-time integration between Flask and React  
✅ High accuracy using Random Forest and XGBoost models  
✅ Responsive, component-based UI built with Tailwind CSS  
✅ Multi-page interface (Home, About) using React Router  
✅ Secure data exchange using Flask-CORS  
✅ Ready for future integration with IoT and weather APIs  

---

## 💻 Technology Stack

### **Frontend**
- **React.js (Vite):** For building fast and modular user interfaces.  
- **Tailwind CSS:** For clean and responsive design.  
- **React Router DOM:** For seamless navigation between pages.  

### **Backend**
- **Flask (Python):** Lightweight backend framework for handling API requests.  
- **Flask-CORS:** Enables secure frontend-backend communication.

### **Machine Learning**
- **Scikit-learn:** For model training, evaluation, and preprocessing.  
- **XGBoost:** Boosted tree-based algorithm providing high performance.  
- **Pandas & NumPy:** For data manipulation and numerical computations.  
- **Pickle:** For serializing the trained model and encoder objects.  

---

## 🧩 Machine Learning Algorithms Used

| Algorithm | Description | Accuracy |
|------------|--------------|-----------|
| **Random Forest** | Ensemble of decision trees for high accuracy and robustness. | ✅ *Best (≈99%)* |
| **SVM** | Finds optimal hyperplane to classify crops based on features. | 97% |
| **KNN** | Classifies based on the nearest data points. | 95% |
| **Decision Tree** | Tree-structured model for interpretability and speed. | 98% |
| **Logistic Regression** | Baseline linear classifier for comparison. | 96% |
| **XGBoost** | Optimized gradient boosting model for high accuracy. | 98% |

---

## 📊 Model Comparison Graph

![Comparison bar](https://github.com/user-attachments/assets/97a27426-23c2-44f7-9799-39e8bf478d43)


---

## 🔄 System Workflow

1. **Input Stage:**  
   User enters N, P, K, temperature, humidity, pH, and rainfall in the web form.

2. **Request Stage:**  
   React sends input data as JSON to Flask backend through an API call.

3. **Prediction Stage:**  
   Flask loads the trained model and predicts the most suitable crop.

4. **Response Stage:**  
   Flask returns a JSON response containing the crop name.

5. **Display Stage:**  
   React dynamically displays the result on the UI.

---

### 🧭 Workflow Diagram

<img width="592" height="410" alt="work flow" src="https://github.com/user-attachments/assets/25829a5c-2e65-440e-ae25-586d45e9c4ca" />

---

---

## Output Screenshots:

<img width="1896" height="827" alt="Screenshot 2025-11-12 104426" src="https://github.com/user-attachments/assets/bf29c754-3869-4fd6-a502-ffcaa43ac1be" />

<img width="1909" height="825" alt="Screenshot 2025-11-12 104638" src="https://github.com/user-attachments/assets/b4bf0550-26ee-4697-bc50-d3fc88d8d665" />

---

---

## 📧 Contact

Email: manthanbhegade407@gmail.com

---


