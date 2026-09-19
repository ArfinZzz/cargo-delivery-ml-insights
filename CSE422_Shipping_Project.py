import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, \
    roc_auc_score
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("CSE422 LAB PROJECT: E-COMMERCE SHIPPING DATASET ANALYSIS")
print("=" * 80)

# ============================================================================
# SECTION 1: LOAD DATASET
# ============================================================================
print("\n[SECTION 1] LOADING DATASET...")
df = pd.read_csv('E_commerce_Shipping_Dataset.csv')

print(f"\nDataset Shape: {df.shape}")
print(f"Total Samples: {df.shape[0]}")
print(f"Total Features: {df.shape[1]}")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

# ============================================================================
# SECTION 2: DATASET DESCRIPTION
# ============================================================================
print("\n" + "=" * 80)
print("[SECTION 2] DATASET DESCRIPTION")
print("=" * 80)

print(f"\n✓ Number of Features: {df.shape[1]}")
print(f"✓ Number of Data Points: {df.shape[0]}")

# Identify feature types
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

print(f"\n✓ Numerical Features ({len(numerical_features)}): {numerical_features}")
print(f"✓ Categorical Features ({len(categorical_features)}): {categorical_features}")

# Identify target variable (usually last column or 'Reached' or 'delivered_on_time')
target_col = df.columns[-1]  # Adjust if needed
print(f"\n✓ Target Variable: {target_col}")
print(f"✓ Unique Classes: {df[target_col].unique()}")
print(f"✓ Problem Type: BINARY CLASSIFICATION (2 classes)")

# ============================================================================
# SECTION 2.1: CORRELATION ANALYSIS
# ============================================================================
print("\n[SECTION 2.1] CORRELATION ANALYSIS")
print("-" * 80)

# Create correlation matrix for numerical features
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('01_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Correlation heatmap saved as: 01_correlation_heatmap.png")

# Key insights from correlation
print("\nKey Insights from Correlation:")
print("- Features with strong positive correlation with target are good predictors")
print("- Features with weak correlation can be dropped for simplification")
print("- Watch for multicollinearity between features")

# ============================================================================
# SECTION 2.2: IMBALANCED DATASET ANALYSIS
# ============================================================================
print("\n[SECTION 2.2] IMBALANCED DATASET ANALYSIS")
print("-" * 80)

class_distribution = df[target_col].value_counts()
print(f"\nClass Distribution:\n{class_distribution}")
print(f"\nClass Percentages:")
for class_val, count in class_distribution.items():
    percentage = (count / len(df)) * 100
    print(f"  Class {class_val}: {count} samples ({percentage:.2f}%)")

# Plot class distribution
plt.figure(figsize=(10, 6))
class_distribution.plot(kind='bar', color=['#FF6B6B', '#4ECDC4'])
plt.title('Class Distribution in Dataset')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.xticks(rotation=0)
for i, v in enumerate(class_distribution):
    plt.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('02_class_imbalance.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Class distribution chart saved as: 02_class_imbalance.png")

# Check if balanced
imbalance_ratio = class_distribution.max() / class_distribution.min()
if imbalance_ratio > 1.5:
    print(f"\n️  Dataset is IMBALANCED (ratio: {imbalance_ratio:.2f})")
    print("   → Use stratified sampling during train-test split")
else:
    print(f"\n✓ Dataset is relatively balanced (ratio: {imbalance_ratio:.2f})")

# ============================================================================
# SECTION 3: DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("[SECTION 3] DATA PREPROCESSING")
print("=" * 80)

# Make a copy for preprocessing
df_processed = df.copy()

# 3.1: Check for missing values
print("\n[SECTION 3.1] HANDLING MISSING VALUES")
print("-" * 80)
missing_values = df_processed.isnull().sum()
print(f"\nMissing Values:\n{missing_values}")

if missing_values.sum() > 0:
    print("\n⚠  Missing values detected!")
    for col in missing_values[missing_values > 0].index:
        if col in numerical_features:
            df_processed[col].fillna(df_processed[col].mean(), inplace=True)
            print(f"   → Filled '{col}' with mean value")
        else:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
            print(f"   → Filled '{col}' with mode value")
    print("✓ Missing values handled!")
else:
    print("✓ No missing values found!")

# 3.2: Handle categorical variables
print("\n[SECTION 3.2] ENCODING CATEGORICAL VARIABLES")
print("-" * 80)

label_encoders = {}
for col in categorical_features:
    if col != target_col:  # Don't encode target yet
        print(f"\nEncoding '{col}':")
        print(f"  Unique values: {df_processed[col].unique()}")
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
        print(f"  ✓ Encoded successfully")

# Encode target variable
print(f"\nEncoding target variable '{target_col}':")
target_encoder = LabelEncoder()
df_processed[target_col] = target_encoder.fit_transform(df_processed[target_col])
print(f"  Mapping: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")
print(f"  ✓ Encoded successfully")

# 3.3: Feature scaling
print("\n[SECTION 3.3] FEATURE SCALING")
print("-" * 80)

X = df_processed.drop(target_col, axis=1)
y = df_processed[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\n✓ Applied StandardScaler to all features")
print(f"  Mean (should be ~0): {X_scaled.mean().mean():.6f}")
print(f"  Std Dev (should be ~1): {X_scaled.std().mean():.6f}")

# ============================================================================
# SECTION 4: DATASET SPLITTING
# ============================================================================
print("\n" + "=" * 80)
print("[SECTION 4] DATASET SPLITTING")
print("=" * 80)

# Use stratified split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Stratified Train-Test Split (80-20):")
print(f"  Training set: {X_train.shape[0]} samples ({(X_train.shape[0] / len(X_scaled)) * 100:.1f}%)")
print(f"  Test set: {X_test.shape[0]} samples ({(X_test.shape[0] / len(X_scaled)) * 100:.1f}%)")
print(f"\n  Class distribution in training set:")
print(f"    {pd.Series(y_train).value_counts().to_dict()}")
print(f"  Class distribution in test set:")
print(f"    {pd.Series(y_test).value_counts().to_dict()}")

# ============================================================================
# SECTION 5: MODEL TRAINING & TESTING (SUPERVISED LEARNING)
# ============================================================================
print("\n" + "=" * 80)
print("[SECTION 5] MODEL TRAINING & TESTING (SUPERVISED LEARNING)")
print("=" * 80)

models = {}
predictions = {}
metrics = {}

# 5.1: K-Nearest Neighbors (KNN)
print("\n[5.1] K-NEAREST NEIGHBORS (KNN)")
print("-" * 80)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
y_pred_proba_knn = knn.predict_proba(X_test)[:, 1]

models['KNN'] = knn
predictions['KNN'] = y_pred_knn
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"✓ KNN trained successfully!")
print(f"  Accuracy: {accuracy_knn:.4f}")

# 5.2: Decision Tree
print("\n[5.2] DECISION TREE")
print("-" * 80)
dt = DecisionTreeClassifier(random_state=42, max_depth=10)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_pred_proba_dt = dt.predict_proba(X_test)[:, 1]

models['Decision Tree'] = dt
predictions['Decision Tree'] = y_pred_dt
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"✓ Decision Tree trained successfully!")
print(f"  Accuracy: {accuracy_dt:.4f}")
print(f"  Feature Importance:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt.feature_importances_
}).sort_values('Importance', ascending=False).head(5)
print(feature_importance.to_string(index=False))

# 5.3: Logistic Regression
print("\n[5.3] LOGISTIC REGRESSION")
print("-" * 80)
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_pred_proba_lr = lr.predict_proba(X_test)[:, 1]

models['Logistic Regression'] = lr
predictions['Logistic Regression'] = y_pred_lr
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"✓ Logistic Regression trained successfully!")
print(f"  Accuracy: {accuracy_lr:.4f}")

# 5.4: Naive Bayes
print("\n[5.4] NAIVE BAYES")
print("-" * 80)
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
y_pred_proba_nb = nb.predict_proba(X_test)[:, 1]

models['Naive Bayes'] = nb
predictions['Naive Bayes'] = y_pred_nb
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"✓ Naive Bayes trained successfully!")
print(f"  Accuracy: {accuracy_nb:.4f}")

# 5.5: Neural Network
print("\n[5.5] NEURAL NETWORK (MLP)")
print("-" * 80)
nn = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
nn.fit(X_train, y_train)
y_pred_nn = nn.predict(X_test)
y_pred_proba_nn = nn.predict_proba(X_test)[:, 1]

models['Neural Network'] = nn
predictions['Neural Network'] = y_pred_nn
accuracy_nn = accuracy_score(y_test, y_pred_nn)
print(f"✓ Neural Network trained successfully!")
print(f"  Accuracy: {accuracy_nn:.4f}")
print(f"  Architecture: Input → 100 neurons → 50 neurons → Output")

# ============================================================================
# SECTION 5.6: UNSUPERVISED LEARNING - K-MEANS CLUSTERING
# ============================================================================
print("\n" + "=" * 80)
print("[SECTION 5.6] UNSUPERVISED LEARNING - K-MEANS CLUSTERING")
print("=" * 80)

print("\nApplying K-Means clustering on entire dataset...")
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

print(f"✓ K-Means clustering completed!")
print(f"  Number of clusters: 2")
print(f"  Cluster distribution: {np.bincount(clusters)}")

# Plot clusters (using 2D PCA for visualization)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, edgecolors='black', linewidth=2, label='Centroids')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
plt.title('K-Means Clustering Results')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.tight_layout()
plt.savefig('03_kmeans_clusters.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ K-Means visualization saved as: 03_kmeans_clusters.png")

# ============================================================================
# SECTION 6: MODEL SELECTION & COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("[SECTION 6] MODEL SELECTION & COMPARISON ANALYSIS")
print("=" * 80)

# Calculate metrics for all models
results = {}
for model_name in predictions.keys():
    y_pred = predictions[model_name]
    y_pred_proba = [y_pred_proba_knn, y_pred_proba_dt, y_pred_proba_lr,
                    y_pred_proba_nb, y_pred_proba_nn]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Confusion Matrix': cm
    }

# 6.1: Accuracy Comparison
print("\n[6.1] ACCURACY COMPARISON")
print("-" * 80)
accuracy_data = {model: results[model]['Accuracy'] for model in results.keys()}
accuracy_df = pd.DataFrame(list(accuracy_data.items()), columns=['Model', 'Accuracy'])
accuracy_df = accuracy_df.sort_values('Accuracy', ascending=False)
print("\nAccuracy Scores:")
print(accuracy_df.to_string(index=False))

plt.figure(figsize=(10, 6))
colors = ['#2ecc71' if x == accuracy_df['Accuracy'].max() else '#3498db'
          for x in accuracy_df['Accuracy']]
bars = plt.bar(accuracy_df['Model'], accuracy_df['Accuracy'], color=colors)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.ylim([0, 1])
for i, (model, acc) in enumerate(zip(accuracy_df['Model'], accuracy_df['Accuracy'])):
    plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('04_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Accuracy comparison chart saved as: 04_accuracy_comparison.png")

# 6.2: Precision & Recall Comparison
print("\n[6.2] PRECISION & RECALL COMPARISON")
print("-" * 80)
precision_data = {model: results[model]['Precision'] for model in results.keys()}
recall_data = {model: results[model]['Recall'] for model in results.keys()}

metrics_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Precision': list(precision_data.values()),
    'Recall': list(recall_data.values())
})
print("\nPrecision & Recall Scores:")
print(metrics_df.to_string(index=False))

x = np.arange(len(metrics_df))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width / 2, metrics_df['Precision'], width, label='Precision', color='#e74c3c')
bars2 = ax.bar(x + width / 2, metrics_df['Recall'], width, label='Recall', color='#3498db')
ax.set_ylabel('Score')
ax.set_title('Precision vs Recall Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
ax.legend()
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig('05_precision_recall_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Precision-Recall comparison chart saved as: 05_precision_recall_comparison.png")

# 6.3: Confusion Matrices
print("\n[6.3] CONFUSION MATRICES")
print("-" * 80)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (model_name, model) in enumerate(models.items()):
    cm = results[model_name]['Confusion Matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
    axes[idx].set_title(f'{model_name}\n(Accuracy: {results[model_name]["Accuracy"]:.4f})')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_xlabel('Predicted')

# Remove extra subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('06_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Confusion matrices saved as: 06_confusion_matrices.png")

print("\nDetailed Confusion Matrices:")
for model_name in results.keys():
    cm = results[model_name]['Confusion Matrix']
    print(f"\n{model_name}:")
    print(f"  True Negatives: {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives: {cm[1, 1]}")

# 6.4: ROC Curves and AUC Scores
print("\n[6.4] ROC CURVES & AUC SCORES")
print("-" * 80)

fig, ax = plt.subplots(figsize=(10, 8))

auc_scores = {}
proba_dict = {
    'KNN': y_pred_proba_knn,
    'Decision Tree': y_pred_proba_dt,
    'Logistic Regression': y_pred_proba_lr,
    'Naive Bayes': y_pred_proba_nb,
    'Neural Network': y_pred_proba_nn
}

for model_name, y_pred_proba in proba_dict.items():
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = auc(fpr, tpr)
    auc_scores[model_name] = auc_score
    ax.plot(fpr, tpr, marker='o', label=f'{model_name} (AUC = {auc_score:.4f})', linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves for All Models')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('07_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAUC Scores:")
auc_df = pd.DataFrame(list(auc_scores.items()), columns=['Model', 'AUC Score'])
auc_df = auc_df.sort_values('AUC Score', ascending=False)
print(auc_df.to_string(index=False))
print("\n✓ ROC curves saved as: 07_roc_curves.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("[FINAL SUMMARY & CONCLUSIONS]")
print("=" * 80)

best_model = accuracy_df.iloc[0]
print(f"\n BEST MODEL: {best_model['Model']}")
print(f"   Accuracy: {best_model['Accuracy']:.4f}")
print(f"   AUC Score: {auc_scores[best_model['Model']]:.4f}")

print("\n  ALL MODELS RANKED BY ACCURACY:")
for idx, row in accuracy_df.iterrows():
    model_name = row['Model']
    acc = row['Accuracy']
    auc = auc_scores[model_name]
    print(f"   {idx + 1}. {model_name:20s} → Accuracy: {acc:.4f} | AUC: {auc:.4f}")

# Histogram analysis for numerical features
plt.figure(figsize=(15, 10))
df[numerical_features].hist(bins=30, figsize=(15, 10), color='#3498db', edgecolor='black')
plt.suptitle('Histogram Analysis of Numerical Features', fontsize=16)
plt.tight_layout()
plt.savefig('08_histograms.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Histogram analysis saved as: 08_histograms.png")

# Box plot analysis for numerical features
plt.figure(figsize=(15, 8))
sns.boxplot(data=df[numerical_features], palette='Set2')
plt.title('Box Plot Analysis of Numerical Features')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('09_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Box plot analysis saved as: 09_boxplots.png")

# Bar chart analysis for categorical features
for col in categorical_features:
    plt.figure(figsize=(10, 6))
    df[col].value_counts().plot(kind='bar', color='#e67e22', edgecolor='black')
    plt.title(f'Bar Chart Analysis of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'10_bar_chart_{col}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Bar chart for '{col}' saved as: 10_bar_chart_{col}.png")
