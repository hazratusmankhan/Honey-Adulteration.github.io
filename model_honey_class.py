import pandas as pd
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle
honey = pd.read_csv("Concentraion_C+Class.csv")
# Check features
features = honey.drop([ "Class", 'Concentration_Class+Class', 'Concentration_Class' ], axis=1)  # Assuming "Class" is the target column
feature_names = features.columns
# Use double square brackets to select a DataFrame with one column
from sklearn.model_selection import train_test_split
y = honey["Class"]
X = honey[feature_names]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save trained model using pickle
pickle.dump(model, open('model_honey_class.pkl', 'wb'))