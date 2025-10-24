import numpy as np
import json
from sklearn.metrics import f1_score, silhouette_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine


class FalsePositiveDetector:
    def __init__(self, embedding_model,
                 auto_multi=True,
                 max_clusters=5,
                 silhouette_threshold=0.25,
                 outlier_clip=True,
                 outlier_alpha=2.5):
        """
        embedding_model: model with .encode(text)
        auto_multi: automatically decide single/multi centroid per field
        max_clusters: upper bound for cluster count
        silhouette_threshold: min silhouette score to justify multiple centroids
        outlier_clip: whether to clip outliers before centroiding
        outlier_alpha: sensitivity for outlier removal (higher = keep more)
        """
        self.model = embedding_model
        self.auto_multi = auto_multi
        self.max_clusters = max_clusters
        self.silhouette_threshold = silhouette_threshold
        self.outlier_clip = outlier_clip
        self.outlier_alpha = outlier_alpha

        self.centroids = {}
        self.thresholds = {}
        self.field_cluster_info = {}

    # ------------------------------------------------
    # 1️⃣ TRAINING PIPELINE
    # ------------------------------------------------
    def train(self, train_data, val_data=None, metric="f1"):
        print("\n=== Building Centroids ===")
        self._build_centroids(train_data)

        print("\n=== Calibrating Thresholds ===")
        self._optimize_thresholds(val_data, metric)

    # ------------------------------------------------
    # 2️⃣ INFERENCE PIPELINE
    # ------------------------------------------------
    def is_false_positive(self, text, field):
        if field not in self.centroids:
            raise ValueError(f"No centroid found for field '{field}'")
        emb = self.model.encode(text)
        emb = emb / np.linalg.norm(emb)
        centroid_obj = np.array(self.centroids[field])
        if centroid_obj.ndim == 2:
            sim = max([1 - cosine(emb, c) for c in centroid_obj])
        else:
            sim = 1 - cosine(emb, centroid_obj)
        th = self.thresholds.get(field, 0.7)
        return 1 if sim < th else 0

    # ------------------------------------------------
    # 3️⃣ SAVE / LOAD MODEL STATE
    # ------------------------------------------------
    def save(self, filepath):
        data = {
            "centroids": {k: np.array(v).tolist() for k, v in self.centroids.items()},
            "thresholds": self.thresholds,
            "field_cluster_info": self.field_cluster_info,
            "config": {
                "auto_multi": self.auto_multi,
                "max_clusters": self.max_clusters,
                "silhouette_threshold": self.silhouette_threshold,
                "outlier_clip": self.outlier_clip,
                "outlier_alpha": self.outlier_alpha,
            }
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Model saved to {filepath}")

    @classmethod
    def load(cls, embedding_model, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        obj = cls(embedding_model,
                  auto_multi=data["config"]["auto_multi"],
                  max_clusters=data["config"]["max_clusters"],
                  silhouette_threshold=data["config"]["silhouette_threshold"],
                  outlier_clip=data["config"]["outlier_clip"],
                  outlier_alpha=data["config"]["outlier_alpha"])
        obj.centroids = {k: np.array(v) for k, v in data["centroids"].items()}
        obj.thresholds = data["thresholds"]
        obj.field_cluster_info = data["field_cluster_info"]
        print(f"✅ Model loaded from {filepath}")
        return obj

    # ------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------
    def _clip_outliers(self, embeddings):
        """Remove embeddings too far from centroid using MAD-based rule."""
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        sims = np.array([1 - cosine(e, centroid) for e in embeddings])
        median = np.median(sims)
        mad = np.median(np.abs(sims - median))
        lower_bound = median - self.outlier_alpha * mad
        mask = sims >= lower_bound
        kept_ratio = np.mean(mask)
        return embeddings[mask], kept_ratio

    def _build_centroids(self, field_text_dict):
        for field, texts in field_text_dict.items():
            embeddings = np.array([self.model.encode(t) for t in texts])
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            # Step 1: Optional outlier clipping
            if self.outlier_clip and len(embeddings) > 5:
                filtered_embs, kept_ratio = self._clip_outliers(embeddings)
                print(f"[{field}] Outlier clipping: kept {kept_ratio*100:.1f}% of points")
                embeddings = filtered_embs

            n = len(embeddings)
            if self.auto_multi and n >= 6:
                best_k, best_score = 1, -1
                for k in range(2, min(self.max_clusters, n - 1) + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    labels = kmeans.fit_predict(embeddings)
                    score = silhouette_score(embeddings, labels)
                    if score > best_score:
                        best_k, best_score = k, score

                if best_score > self.silhouette_threshold:
                    kmeans = KMeans(n_clusters=best_k, random_state=42)
                    kmeans.fit(embeddings)
                    centroids = kmeans.cluster_centers_
                    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
                    self.centroids[field] = centroids
                    self.field_cluster_info[field] = {
                        "mode": "multi",
                        "clusters": best_k,
                        "silhouette": best_score
                    }
                    print(f"[{field}] multi-centroid ({best_k} clusters, silhouette={best_score:.2f})")
                    continue

            # fallback single centroid
            centroid = np.mean(embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            self.centroids[field] = centroid
            self.field_cluster_info[field] = {"mode": "single", "clusters": 1, "silhouette": None}
            print(f"[{field}] single centroid")

    def _optimize_thresholds(self, validation_data=None, metric="f1"):
        for field, centroid_obj in self.centroids.items():
            if validation_data and field in validation_data:
                samples = validation_data[field]
                sims, labels = [], []
                for text, label in samples:
                    emb = self.model.encode(text)
                    emb = emb / np.linalg.norm(emb)
                    if isinstance(centroid_obj, np.ndarray) and centroid_obj.ndim == 2:
                        sim = max([1 - cosine(emb, c) for c in centroid_obj])
                    else:
                        sim = 1 - cosine(emb, centroid_obj)
                    sims.append(sim)
                    labels.append(label)

                sims, labels = np.array(sims), np.array(labels)
                thresholds = np.linspace(0.3, 0.95, 100)
                best_th, best_score = 0.7, -1
                for th in thresholds:
                    preds = (sims >= th).astype(int)
                    score = f1_score(labels, preds)
                    if score > best_score:
                        best_score, best_th = score, th

                self.thresholds[field] = best_th
                print(f"[{field}] optimized threshold={best_th:.3f} (F1={best_score:.3f})")
            else:
                self.thresholds[field] = 0.7
                print(f"[{field}] default threshold=0.7 (no validation data)")


## training calls

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

train_data = {
    "city": ["London", "Paris", "New York", "Tokyo", "Delhi", "Bangalore", "Mumbai", "Car", "Train"],
    "name": ["Alice", "Bob", "Charlie", "David", "Eve"]
}

val_data = {
    "city": [("London", 1), ("NYC", 1), ("Car", 0), ("Bus", 0)],
    "name": [("Alice", 1), ("Random", 0)]
}

detector = FalsePositiveDetector(model)
detector.train(train_data, val_data)
detector.save("false_positive_model.json")

## inference calls

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
detector = FalsePositiveDetector.load(model, "false_positive_model.json")

print(detector.is_false_positive("Bus", "city"))      # 1 → false positive
print(detector.is_false_positive("Paris", "city"))    # 0 → true positive
