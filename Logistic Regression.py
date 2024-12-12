import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import mlflow

# File path to your dataset
file_path = r"C:\Users\e\PycharmProjects\pythonProject\cleaned_combined_arem_dataset.csv"

# 1. Load dataset and fix headers
columns = ['timestamp', 'x-axis', 'y-axis', 'z-axis', 'other_features', 'activity_label']
df = pd.read_csv(file_path, header=None, names=columns)

# 2. Clean the dataset
# a) Remove rows with non-numeric features
for col in ['x-axis', 'y-axis', 'z-axis', 'other_features']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# b) Drop rows with NaN values
df.dropna(inplace=True)

# c) Display dataset summary
print("Dataset after cleaning:")
print(df.head())
print("Dataset shape:", df.shape)
print(df.info())

# 3. Prepare Features and Target
X = df[['x-axis', 'y-axis', 'z-axis', 'other_features']]  # Select feature columns
y = df['activity_label']  # Target column

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Normalize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Train Logistic Regression
log_reg = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs', multi_class='multinomial')
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# Evaluate performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Log metrics and results with MLflow
mlflow.set_experiment("Human Activity Recognition - Specific Axes")

with mlflow.start_run():
    # Log model parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", log_reg.max_iter)
    mlflow.log_param("solver", log_reg.solver)

    # Log metrics
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # Save classification report as an artifact
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = "classification_report_logistic_regression.csv"
    report_df.to_csv(report_path)
    mlflow.log_artifact(report_path)

print(f"Accuracy: {accuracy:.2f}")
