import pandas as pd 
from sklearn.model_selection import train_test_split 
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
# Load dataset
df = pd.read_csv("data/cleaned_ai4i2020.csv")
cols=df[['Power','OSF','PWF','HDF','TWF','Torque [Nm]','Rotational speed [rpm]','Temp_Difference']]

# Define target variable
y = df['Machine failure']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    cols, y, test_size=0.2, random_state=42, stratify=y
)
best_params = {'learning_rate': 0.1, 'l2_leaf_reg': 5, 'iterations': 300, 'depth': 6, 'border_count': 128}
model = CatBoostClassifier(random_state=42, **best_params)
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', model)
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(pipeline,"app/models/catboost_smote_pipeline.joblib")

print("Training complete. Model saved")