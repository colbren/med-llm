import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Load the dataset while preserving the header
df = pd.read_csv('TCGA-LUAD_final.txt', sep='\t')

# Drop any rows with missing values (if needed)
df = df.dropna()

# Define metadata and feature columns
metadata_cols = df.columns[:2]      # "Sample Type" and "File ID"
feature_cols  = df.columns[2:]       # All gene expression features (8000+)

# Separate labels and features
y = df.iloc[:, 0]       # "Sample Type"
X = df.iloc[:, 2:]      # Gene expression features

# Encode the labels (needed for mutual information calculation)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normalize the gene expression features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Shape of normalized features:", X_scaled.shape)

# Use SelectKBest to select the top 1000 features based on mutual information
selector = SelectKBest(mutual_info_classif, k=1000)
X_kbest = selector.fit_transform(X_scaled, y_encoded)
print("Shape after SelectKBest:", X_kbest.shape)

# Get the indices of the selected features, then map them back to the original column names
selected_indices = selector.get_support(indices=True)
selected_feature_names = feature_cols[selected_indices]

# Create the final DataFrame by combining the metadata with the selected features
final_df = pd.concat([df[metadata_cols], df[selected_feature_names]], axis=1)

# Save the resulting DataFrame to a file with the same tab-delimited format
final_df.to_csv('Top1000_Features.txt', sep='\t', index=False)
print("✅ Top 1000 features saved to 'Top1000_Features.txt'!")
