#LOAD DATASET

import pandas as pd

df = pd.read_csv(r"/Users/sauravkumararya/Documents/GitHub/AI_HEART_PREDICTION/Dataset/Dataset.csv")
print(df.head())

#CHECK DATASET INFO NAD MISSING VALUE

print(df.info())
print(df.isnull().sum())

#CONVERT OBJECT COLUMNS TO NUMERIC

df = df.apply(pd.to_numeric,errors='coerce')
print(df.info())
print(df.head())

#CHECK FOR ANY NAN VALUES AGAIN

print("\nmissing value after conversion:\n",df.isnull().sum())

#ENCODE CATEGORICAL VARIABLE

from sklearn.preprocessing import LabelEncoder
categorical_cols = ["sex", "cp", "thal", "restecg" , "slope", "age", "trestbps", "chol", "fbs", "thalach", "exang", "oldpeak", "ca"]
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])
print(df.head())

#CORRELATION PLOT OF ALL DATA

import matplotlib.pyplot as plt
import seaborn as sns
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Plot")
plt.show()

#SPLIT DATA INTO TRAIN AND TEST SETS

import pandas as pd

# Load training data
train_df = pd.read_csv(r"/Users/sauravkumararya/Documents/GitHub/AI_HEART_PREDICTION/Dataset/heart_train (1).csv")
X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

# Load testing data
test_df = pd.read_csv(r"/Users/sauravkumararya/Documents/GitHub/AI_HEART_PREDICTION/Dataset/heart_test (1).csv")
X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]


#TRAIN AND EVALUATE EACH MODEL
#LOGISTIC REGRESSION

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logreg = LogisticRegression(max_iter=1000)  
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

# Accuracy
print('Accuracy of the Logistic Regression model is =', accuracy_score(y_test, y_pred))

# Classification report
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data=cm, 
                           columns=['Predicted:0', 'Predicted:1'], 
                           index=['Actual:0', 'Actual:1'])

# Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Greens")
plt.title("Logistic Regression - Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

#ROC CURVE
from sklearn.metrics import roc_curve, roc_auc_score
y_probs = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


#SUPPORT VECTOR CLASSIFIER(SVM)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)


y_pred_svm = svm_model.predict(X_test)

# Accuracy
print('Accuracy of SVM model =', accuracy_score(y_test, y_pred_svm))

# Classification report
print('The details for confusion matrix is =')
print(classification_report(y_test, y_pred_svm))

# Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
conf_matrix_svm = pd.DataFrame(cm_svm,
                               columns=['Predicted:0', 'Predicted:1'],
                               index=['Actual:0', 'Actual:1'])

plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap="Blues")
plt.title("SVM Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

#ROC CURVE

y_probs_svm = svm_model.predict_proba(X_test)[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_probs_svm)
auc_svm = roc_auc_score(y_test, y_probs_svm)
plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.2f})', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


#RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# Accuracy
print('Accuracy of Random Forest model =', accuracy_score(y_test, y_pred_rf))

# Classification Report
print('The details for confusion matrix is =')
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
conf_matrix_rf = pd.DataFrame(cm_rf, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])

plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap="Oranges")
plt.title("Random Forest Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

#ROC CURVE

y_probs_rf = rf_model.predict_proba(X_test)[:, 1]

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_probs_rf)

auc_rf = roc_auc_score(y_test, y_probs_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})', color='darkgreen')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


#DECISION TREE

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

print("\n--- Decision Tree Classifier Model ---")

dt_model = DecisionTreeClassifier(random_state=42) 
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print('Accuracy of Decision Tree model =', accuracy_score(y_test, y_pred_dt))
print('The details for confusion matrix is =')
print(classification_report(y_test, y_pred_dt))

cm_dt = confusion_matrix(y_test, y_pred_dt)
conf_matrix_dt = pd.DataFrame(cm_dt, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])

plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap="Reds")
plt.title("Decision Tree Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

#ROC CURVE

y_probs_dt = dt_model.predict_proba(X_test)[:, 1]

fpr_dt, tpr_dt, _ = roc_curve(y_test, y_probs_dt)

auc_dt = roc_auc_score(y_test, y_probs_dt)

plt.figure(figsize=(8, 6))
plt.plot(fpr_dt, tpr_dt, color='red', label=f'Decision Tree (AUC = {auc_dt:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()



#K-NEAREST NEIGHBORS

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

# Accuracy and Classification Report
print('Accuracy of KNN model =', accuracy_score(y_test, y_pred_knn))
print('Classification Report:')
print(classification_report(y_test, y_pred_knn))

# Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
conf_matrix_knn = pd.DataFrame(cm_knn, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])

plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap="Purples")
plt.title("KNN Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

#ROC CURVE

y_probs_knn = knn_model.predict_proba(X_test)[:, 1]

fpr_knn, tpr_knn, _ = roc_curve(y_test, y_probs_knn)
auc_knn = roc_auc_score(y_test, y_probs_knn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {auc_knn:.2f})', color='purple')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()



#ENSAMBLE MODEL

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Train base models
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)

dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Get training meta-features
dt_preds = dt_model.predict_proba(X_train)
rf_preds = rf_model.predict_proba(X_train)
meta_features = np.hstack((dt_preds, rf_preds))

# Train XGBoost as meta-model
xgb_meta_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_meta_model.fit(meta_features, y_train)

#Prepare test meta-features
dt_test_preds = dt_model.predict_proba(X_test)
rf_test_preds = rf_model.predict_proba(X_test)
meta_test_features = np.hstack((dt_test_preds, rf_test_preds))

#Predict and evaluate
final_preds = xgb_meta_model.predict(meta_test_features)
final_probs = xgb_meta_model.predict_proba(meta_test_features)[:, 1] 

print("Stacked Boosted Model Accuracy:", accuracy_score(y_test, final_preds))
print("Classification Report:\n", classification_report(y_test, final_preds))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, final_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Stacked Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


fpr_lr, tpr_lr, _ = roc_curve(y_test, y_probs)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_probs_svm)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_probs_knn)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_probs_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_probs_rf)

auc_lr = auc(fpr_lr, tpr_lr)
auc_svm = auc(fpr_svm, tpr_svm)
auc_knn = auc(fpr_knn, tpr_knn)
auc_tree = auc(fpr_dt, tpr_dt)
auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(10, 7))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.2f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.2f})')
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {auc_knn:.2f})')
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison of ML Models')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()




#SAVE TRAINED MODEL

import pickle  

models = {
    "Logistic Regression": logreg,
    "Random Forest": rf_model,
    "K-Nearest Neighbors": knn_model,
    "Decision Tree": dt_model,
    "Support Vector Machine": svm_model
}

with open("trained_models.pkl", "wb") as file:
    pickle.dump(models, file)
print("Models saved successfully!")

# Load the trained models
with open("trained_models.pkl", "rb") as file:
    loaded_models = pickle.load(file)
print(loaded_models)
print("Model loaded successfully")