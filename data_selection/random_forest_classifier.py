import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Step 1: Load the Selected Features File
# -------------------------------
# The file Top1000_Features.txt should have the same format as your original file:
# First column: "Sample Type", second column: "File ID", then 1000 gene expression features.
final_df = pd.read_csv('Top1000_Features.txt', sep='\t')

# Preview the first few rows to verify the structure
print("Preview of Top1000_Features.txt:")
print(final_df.head())

# -------------------------------
# Step 2: Prepare Data for Modeling
# -------------------------------
# Extract metadata and features
# Metadata: "Sample Type" (label) and "File ID" (identifier)
metadata_cols = final_df.columns[:2]  # "Sample Type" and "File ID"
feature_cols  = final_df.columns[2:]   # The 1000 selected gene expression features

# Set X as the gene expression features and y as the sample type label
X = final_df[feature_cols]
y = final_df.iloc[:, 0]  # "Sample Type"

# Encode the labels if they are not numeric (e.g., "Primary Tumor", "Solid Tissue Normal")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -------------------------------
# Step 3: Split Data into Training and Testing Sets
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# -------------------------------
# Step 4: Train the Random Forest Classifier
# -------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# -------------------------------
# Step 5: Evaluate the Model using AUC
# -------------------------------
# For binary classification, we use the probability of the positive class.
# If you have multi-class, you'll need to adjust this (e.g., one-vs-rest approach).
y_probs = rf_model.predict_proba(X_test)

# Calculate AUC score
auc_score = roc_auc_score(y_test, y_probs, multi_class='ovr')
print("AUC Score:", auc_score)

if auc_score > 0.8:
    print("Selected features are effective (AUC > 0.8).")
else:
    print("Selected features are not effective (AUC <= 0.8).")
