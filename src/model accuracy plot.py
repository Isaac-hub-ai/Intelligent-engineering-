import matplotlib.pyplot as plt
import numpy as np

# Example accuracy scores for different models
models = ["Logistic Regression", "Random Forest", "SVM", "KNN"]
accuracies = [0.85, 0.90, 0.88, 0.84]  # Replace with actual values from your models

# Plot accuracies
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'red'])
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0.7, 1.0)  # Adjust y-axis range as needed
plt.show()
