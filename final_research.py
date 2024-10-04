!pip install fairlearn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate, equalized_odds_difference

data = pd.read_csv("/content/recruitment_data.csv")

# Display the first few rows of the dataset
print(data.head())

# Prepare data
X = data.drop('HiringDecision', axis=1)
y = data['HiringDecision']
sensitive_feature = data['Gender']  # Use 'Gender' as the sensitive attribute

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X, y, sensitive_feature, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)  # Enable probability estimates
}

results = {}

# Train models and evaluate them
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability of the positive class

    # Calculate fairness metrics
    metric_frame = MetricFrame(
        metrics={
            'accuracy': accuracy_score,
            'true_positive_rate': true_positive_rate,
            'false_positive_rate': false_positive_rate
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_test
    )

    equalized_odds_diff = equalized_odds_difference(y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test)

    results[name] = {
        'Accuracy': metric_frame.overall['accuracy'],
        'Equalized Odds Difference': equalized_odds_diff
    }

# Display the results
for model, metrics in results.items():
    print(f"Model: {model}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Equalized Odds Difference: {metrics['Equalized Odds Difference']:.4f}\n")

# Find the model with the Equalized Odds Difference closest to zero
best_model = min(results, key=lambda x: abs(results[x]['Equalized Odds Difference']))
best_model_accuracy = results[best_model]['Accuracy']
best_model_eo_diff = results[best_model]['Equalized Odds Difference']

print(f"The best model is {best_model} with an Accuracy of {best_model_accuracy:.4f} and an Equalized Odds Difference of {best_model_eo_diff:.4f}")

# Import the joblib library
import joblib

# Save the best model for deployment
joblib.dump(models[best_model], 'best_hiring_model_enhanced_v7.pkl')

# Load the model and make predictions on new applicants
# Creating a sample new applicant data
new_applicants = pd.DataFrame({
    'Age': [30, 25, 40],
    'Gender': [1, 0, 1],
    'EducationLevel': [3, 2, 4],
    'ExperienceYears': [5, 3, 10],
    'PreviousCompanies': [2, 1, 4],
    'DistanceFromCompany': [15.0, 20.0, 5.0],
    'InterviewScore': [85, 70, 90],
    'SkillScore': [80, 75, 95],
    'PersonalityScore': [70, 60, 85],
    'RecruitmentStrategy': [1, 2, 3]
})

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Recreate the label encoders 
label_encoders = {}
for col in ['Gender', 'EducationLevel', 'RecruitmentStrategy']:  # Add other categorical columns as needed
    le = LabelEncoder()
    # You'll need to fit the encoders on the original data used for training
    # For demonstration, I'm fitting on the 'new_applicants' DataFrame,
    # but you should replace this with the appropriate training data
    le.fit(new_applicants[col])
    label_encoders[col] = le

# Transform the new applicant data using the label encoders
new_applicants_transformed = new_applicants.apply(lambda col: label_encoders[col.name].transform(col) if col.name in label_encoders else col)
loaded_model = joblib.load('best_hiring_model_enhanced_v7.pkl')
predictions = loaded_model.predict(new_applicants_transformed)
new_applicants['hiring_decision'] = predictions

print(new_applicants)

import matplotlib.pyplot as plt
import seaborn as sns

# ... (preceding code) ...

# Visualize results
# 1. Accuracy Comparison
plt.figure(figsize=(6, 4))
sns.barplot(x=list(results.keys()), y=[result['Accuracy'] for result in results.values()])
plt.title('Accuracy Comparison of Models')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Equalized Odds Difference Comparison
plt.figure(figsize=(6, 4))
sns.barplot(x=list(results.keys()), y=[result['Equalized Odds Difference'] for result in results.values()])
plt.title('Equalized Odds Difference Comparison of Models')
plt.ylabel('Equalized Odds Difference')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Distribution of Predictions by Gender (for the best model)
best_model_predictions = models[best_model].predict(X_test)
plt.figure(figsize=(6,4))
sns.countplot(x=best_model_predictions, hue=sensitive_test)
plt.title(f'Distribution of Predictions by Gender ({best_model})')
plt.xlabel('Hiring Decision (0: No, 1: Yes)')
plt.ylabel('Count')
plt.show()
