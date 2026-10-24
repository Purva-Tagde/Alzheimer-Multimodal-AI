from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import joblib


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
TRAIN_DIR = Path("data/processed/mri_images/train")
TEST_DIR = Path("data/processed/mri_images/test")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------
device = torch.device("cpu")
print("Device:", device)


# ---------------------------------------------------------
# Load pretrained ResNet18
# ---------------------------------------------------------
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Remove final classification layer
feature_extractor = nn.Sequential(*list(model.children())[:-1])

feature_extractor.eval()
feature_extractor.to(device)

print("ResNet18 feature extractor loaded.")


# ---------------------------------------------------------
# Convert 5 grayscale slices into 5 RGB-like inputs
# ---------------------------------------------------------
def extract_features(path):
    x = np.load(path).astype(np.float32)

    # Expected shape: (5, 224, 224)
    if x.shape != (5, 224, 224):
        raise ValueError(f"Unexpected shape {x.shape} in {path.name}")

    # Normalize 0-255 -> 0-1
    x = x / 255.0

    # Each slice becomes 3-channel grayscale
    # Shape: (5, 3, 224, 224)
    x = np.repeat(x[:, None, :, :], 3, axis=1)

    tensor = torch.from_numpy(x).to(device)

    with torch.no_grad():
        features = feature_extractor(tensor)

    # (5, 512, 1, 1) -> (5, 512)
    features = features.squeeze(-1).squeeze(-1)

    # Average the 5 slices
    features = features.mean(dim=0)

    return features.cpu().numpy()


# ---------------------------------------------------------
# Extract dataset
# ---------------------------------------------------------
def load_dataset(directory):
    X = []
    y = []
    names = []

    files = sorted(directory.glob("*.npy"))

    for i, path in enumerate(files, start=1):
        print(f"Processing {i}/{len(files)}: {path.name}")

        X.append(extract_features(path))

        # Label comes from filename
        # Example: 011_S_0003_I7055_AD.npy
        label = 1 if path.stem.endswith("_AD") else 0

        y.append(label)
        names.append(path.name)

    return np.asarray(X), np.asarray(y), names


print("\nExtracting TRAIN features...")
X_train, y_train, train_names = load_dataset(TRAIN_DIR)

print("\nExtracting TEST features...")
X_test, y_test, test_names = load_dataset(TEST_DIR)


print("\nFeature shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# ---------------------------------------------------------
# Save extracted features
# ---------------------------------------------------------
np.save("data/processed/mri_resnet_train_features.npy", X_train)
np.save("data/processed/mri_resnet_test_features.npy", X_test)

joblib.dump(
    {
        "train_names": train_names,
        "test_names": test_names,
    },
    "data/processed/mri_resnet_feature_metadata.pkl",
)


# ---------------------------------------------------------
# Logistic Regression classifier
# ---------------------------------------------------------
classifier = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    random_state=42,
)

classifier.fit(X_train, y_train)

probabilities = classifier.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
specificity = recall_score(
    y_test,
    predictions,
    pos_label=0,
    zero_division=0,
)
f1 = f1_score(y_test, predictions, zero_division=0)
auc = roc_auc_score(y_test, probabilities)

cm = confusion_matrix(y_test, predictions)


print("\n" + "=" * 60)
print("MRI TRANSFER LEARNING + LOGISTIC REGRESSION")
print("=" * 60)

print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"ROC-AUC      : {auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["CN", "AD"],
        zero_division=0,
    )
)


# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------
joblib.dump(
    classifier,
    MODEL_DIR / "mri_resnet18_feature_logistic.pkl",
)

print("\nSaved:")
print(MODEL_DIR / "mri_resnet18_feature_logistic.pkl")
print("Experiment complete.")