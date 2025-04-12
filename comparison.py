import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Load the dataset
data = pd.read_csv('crop_data.csv')

# Prepare the data (features and target)
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Encode the target labels
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# Store results
metrics = {}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    metrics[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1-score": f1_score(y_test, y_pred, average='weighted')
    }

# Print results
metrics_df = pd.DataFrame(metrics).T
print(metrics_df)

# Save the best model
best_model_name = metrics_df['F1-score'].idxmax()
best_model = models[best_model_name]

# Save models and preprocessing objects
with open('model/crop_recommendation_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)
with open('model/preprocessing.pkl', 'wb') as preprocessing_file:
    pickle.dump(scaler, preprocessing_file)
with open('model/label_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(y_encoder, encoder_file)

print(f"Best Model: {best_model_name} with F1-score: {metrics_df.loc[best_model_name, 'F1-score']:.2f}")
print("Model, scaler, and label encoder saved successfully.")
