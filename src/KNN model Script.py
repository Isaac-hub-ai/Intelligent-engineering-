import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import mlflow

# File path to your dataset
file_path = r"C:\Users\e\PycharmProjects\pythonProject\cleaned_combined_arem_dataset.csv"


# 1. Load the dataset assuming specific axis columns and other relevant data
columns = ['x-axis', 'y-axis', 'z-axis', 'other_features', 'activity_label']  # Example column names
df = pd.read_csv(file_path, names=columns)

# 2. Data Cleaning
# a) Strip whitespace from all string entries
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# b) Handle missing values by forward filling
df.fillna(method='ffill', inplace=True)

# Display basic info about the dataset after cleaning
print("Dataset after cleaning:")
print(df.head())
print("Dataset shape:", df.shape)
print(df.info())

# 3. Separate Features and Target
X = df[['x-axis', 'y-axis', 'z-axis', 'other_features']]  # Select specific columns for features
y = df['activity_label']  # Use the activity label column as the target

# Note: Ensure non-numeric columns in X are properly encoded or excluded
X = pd.get_dummies(X, drop_first=True)

# Encode the target labels into integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Normalize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Train and Evaluate the KNN Model
k_neighbors = 5  # You can adjust this value to tune the model
knn = KNeighborsClassifier(n_neighbors=k_neighbors)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Evaluate performance
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Log metrics and results with MLflow
mlflow.set_experiment("Human Activity Recognition - KNN Model")

with mlflow.start_run():
    # Log model parameters
    mlflow.log_param("model_type", "KNN")
    mlflow.log_param("n_neighbors", knn.n_neighbors)

    # Log metrics
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # Save classification report as an artifact
    report = classification_report(y_test, y_pred, zero_division=1, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = "classification_report.csv"
    report_df.to_csv(report_path)
    mlflow.log_artifact(report_path)

print(f"Accuracy: {accuracy:.2f}")
