import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import skeletonize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import yaml

SUPPORTED_EXTENSIONS = {".bmp"}


def extract_glcm_features(image_gray):
    img_64 = np.clip(image_gray // 4, 0, 63).astype(np.uint8)
    glcm = graycomatrix(
        img_64,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=64,
        symmetric=True,
        normed=True,
    )

    return {
        "contrast": graycoprops(glcm, "contrast").mean(),
        "dissimilarity": graycoprops(glcm, "dissimilarity").mean(),
        "homogeneity": graycoprops(glcm, "homogeneity").mean(),
        "energy": graycoprops(glcm, "energy").mean(),
        "correlation": graycoprops(glcm, "correlation").mean(),
    }


def get_fractal_dimension(image_gray):
    _, binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = (binary == 0).astype(np.uint8)

    p = int(np.floor(np.log2(min(binary.shape))))
    pixels = binary[: 2**p, : 2**p]

    def count(pixels, k):
        S = np.add.reduceat(
            np.add.reduceat(pixels, np.arange(0, pixels.shape[0], k), axis=0),
            np.arange(0, pixels.shape[1], k),
            axis=1,
        )
        return np.count_nonzero(S)

    scales = 2 ** np.arange(p, 1, -1)
    counts = np.array([count(pixels, s) for s in scales], dtype=float)

    return float(np.polyfit(np.log(1.0 / scales), np.log(counts), 1)[0])


def extract_morphology_features(image_gray):
    _, binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = (binary > 0).astype(np.uint8)
    skeleton = skeletonize(binary.astype(bool)).astype(np.uint8)

    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    filtered = cv2.filter2D(skeleton, -1, kernel)
    branch_points = int(np.sum((skeleton == 1) & (filtered >= 13)))

    density = float(np.count_nonzero(binary) / binary.size)
    return {"branch_count": branch_points, "fern_density": density}


def extract_fft_features(image_gray):
    dft = np.fft.fft2(image_gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.log1p(np.abs(dft_shift))

    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2
    radius = 30

    y, x = np.ogrid[-center_h : h - center_h, -center_w : w - center_w]
    mask = x * x + y * y <= radius * radius

    low_freq_mean = float(np.mean(magnitude_spectrum[mask]))
    high_freq_mean = float(np.mean(magnitude_spectrum[~mask]))
    return {"fft_low_freq": low_freq_mean, "fft_high_freq": high_freq_mean}


def image_to_feature_vector(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

    features = {}
    features.update(extract_glcm_features(img))
    features.update(extract_fft_features(img))
    features.update(extract_morphology_features(img))
    features["fractal_dim"] = get_fractal_dimension(img)
    return features


def load_dataset(imgs_dir, labels_dir):
    rows = []
    imgs_dir = Path(imgs_dir)
    labels_dir = Path(labels_dir)

    for img_path in imgs_dir.iterdir():
        if not img_path.stem.endswith("_0"):
            continue
        label_path = labels_dir / (img_path.stem + ".txt") 
        rows.append({**image_to_feature_vector(img_path), "class": label_path.read_text().strip()})

    return pd.DataFrame(rows)


def preprocess_data(df, test_size=0.2):
    X = df.drop("class", axis=1)
    y = LabelEncoder().fit_transform(df["class"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_tear_classifier(df, test_size=0.2):
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df, test_size)
    model = RandomForestClassifier(n_estimators=10, class_weight="balanced", min_samples_split=5)
    
    # Add CV score
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1_macro')
    print(f"CV F1 Scores: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_test_scaled, y_test


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Which features are most diagnostic?")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()


def plot_umap_projection(df):
    features = df.drop("class", axis=1)
    scaled_data = StandardScaler().fit_transform(features)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(scaled_data)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=df["class"],
        palette="Spectral",
        s=60,
        alpha=0.8,
    )
    plt.title("UMAP Projection: How distinct are the disease clusters?")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(df):
    feature_names = [col for col in df.columns if col != "class"]
    rows = int(np.ceil(len(feature_names) / 5))
    fig, axes = plt.subplots(rows, 5, figsize=(22, 4.5 * rows), squeeze=False)

    palette = sns.color_palette("husl", len(df["class"].unique()))

    for i, feature in enumerate(feature_names):
        ax = axes[i // 5][i % 5]
        sns.stripplot(
            data=df,
            x="class",
            y=feature,
            ax=ax,
            hue="class",
            palette=palette,
            jitter=0.2,
            size=4,
            alpha=0.7,
            legend=False,
        )
        ax.set_title(feature, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

    for empty_ax in axes.flat[len(feature_names) :]:
        empty_ax.axis("off")

    plt.tight_layout()
    plt.suptitle("Feature Distributions by Class", fontsize=18, y=1.02)
    plt.show()

def load_data(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    imgs_dir = config["imgs_path"]
    labels_dir = config["labels_path"]
    models_dir = config["models_dir"]
    return load_dataset(imgs_dir, labels_dir), models_dir

def train_model(model_path, df):
    model, scaler, X_test, y_test = train_tear_classifier(df)

    y_pred = model.predict(X_test)
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    data = {'model': model, 'scaler': scaler}
    pickle.dump(data, open(model_path, "wb"))

def test_model(model_path, df):
    from cl_inference import TearClassifier
    classifier = TearClassifier(model_path)
    
    X = df.drop("class", axis=1)
    y = LabelEncoder().fit_transform(df["class"])
    X_scaled = classifier.scaler.transform(X)
    
    y_pred = classifier.model.predict(X_scaled)
    print("--- Classification Report ---")
    print(classification_report(y, y_pred))
    plot_feature_importance(classifier.model, df.drop("class", axis=1).columns)
    plot_umap_projection(df)
    plot_feature_distributions(df)


if __name__ == "__main__":
    config_path = Path("src/models/config/cl_models/config.yaml")
    df, models_dir = load_data(config_path)
    model_path = Path(models_dir) / "tear_classifier.pkl"
    if model_path.exists():
        print(f"Model already exists at {model_path}. Loading and testing...")
        test_model(model_path, df)
    else:
        train_model(model_path, df)
        test_model(model_path, df)