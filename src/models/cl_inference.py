import os
import pickle
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import skeletonize


class TearClassifier:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        data = pickle.load(open(model_path, "rb"))
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 
                              'fft_low_freq', 'fft_high_freq', 'branch_count', 'fern_density', 'fractal_dim']

    def extract_glcm_features(self, image_gray):
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

    def get_fractal_dimension(self, image_gray):
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

    def extract_morphology_features(self, image_gray):
        _, binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = (binary > 0).astype(np.uint8)
        skeleton = skeletonize(binary.astype(bool)).astype(np.uint8)

        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
        filtered = cv2.filter2D(skeleton, -1, kernel)
        branch_points = int(np.sum((skeleton == 1) & (filtered >= 13)))

        density = float(np.count_nonzero(binary) / binary.size)
        return {"branch_count": branch_points, "fern_density": density}

    def extract_fft_features(self, image_gray):
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

    def image_to_feature_vector(self, image_path):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

        features = {}
        features.update(self.extract_glcm_features(img))
        features.update(self.extract_fft_features(img))
        features.update(self.extract_morphology_features(img))
        features["fractal_dim"] = self.get_fractal_dimension(img)
        return [features[name] for name in self.feature_names]

    def predict(self, image_path):
        features = self.image_to_feature_vector(image_path)
        features_scaled = self.scaler.transform([features])
        return self.model.predict(features_scaled)[0]

    def predict_proba(self, image_path):
        features = self.image_to_feature_vector(image_path)
        features_scaled = self.scaler.transform([features])
        return self.model.predict_proba(features_scaled)[0]
