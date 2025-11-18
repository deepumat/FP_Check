"""
PCA scatter of word embeddings with interactive hover showing the word and centroid.
Requirements:
    pip install numpy matplotlib scikit-learn
Run in a Python environment with a GUI/interactive matplotlib backend (Jupyter, or python with TkAgg/Qt5Agg).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_embeddings_pca(word_embeddings,
                        title="PCA of word embeddings",
                        marker_size=40,
                        centroid_marker_size=140,
                        hover_dist_thresh=0.04):
    """
    Plot 2D PCA projection of embeddings with interactive hover to show the word.
    Args:
        word_embeddings: dict mapping word (str) -> 1D array-like embedding (all same length)
        title: plot title
        marker_size: scatter marker size for points
        centroid_marker_size: marker size for centroid
        hover_dist_thresh: fraction of axis span used to decide hover sensitivity (0..1)
    Returns:
        dict with keys: "pca" (fitted PCA object), "coords" (n x 2 array), "words" (list), "centroid" (1x2)
    """
    # --- prepare data
    if not isinstance(word_embeddings, dict) or len(word_embeddings) == 0:
        raise ValueError("word_embeddings must be a non-empty dict {word: embedding_vector}")

    words = list(word_embeddings.keys())
    X = np.vstack([np.asarray(word_embeddings[w]) for w in words])
    if X.ndim != 2:
        raise ValueError("embeddings should be 1D vectors of equal length")

    # --- PCA to 2D
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)  # shape (n_words, 2)

    # centroid in PCA space
    centroid = X2.mean(axis=0)

    # --- plot
    fig, ax = plt.subplots(figsize=(9, 6))
    pts = ax.scatter(X2[:, 0], X2[:, 1], s=marker_size, picker=True)
    # centroid marker (distinct)
    ax.scatter([centroid[0]], [centroid[1]], s=centroid_marker_size, marker='X', label='centroid')
    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True, linestyle='--', alpha=0.4)

    # annotation used for hover
    annot = ax.annotate("", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    xdata = X2[:, 0]
    ydata = X2[:, 1]

    def update_annot(i):
        pos = (xdata[i], ydata[i])
        annot.xy = pos
        annot.set_text(words[i])
        annot.get_bbox_patch().set_alpha(0.9)

    def on_move(event):
        if event.inaxes != ax:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return

        # compute distances in data coords
        dx = xdata - event.xdata
        dy = ydata - event.ydata
        dist = np.hypot(dx, dy)

        # threshold in data units: use axis span * hover_dist_thresh
        xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
        yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
        thresh = hover_dist_thresh * max(xspan, yspan)

        nearest = np.argmin(dist)
        if dist[nearest] < thresh:
            update_annot(nearest)
            if not annot.get_visible():
                annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

    # Optional: clicking can also print or return the word (example)
    def on_click(event):
        if event.inaxes != ax:
            return
        # find nearest point
        dx = xdata - event.xdata
        dy = ydata - event.ydata
        dist = np.hypot(dx, dy)
        nearest = np.argmin(dist)
        xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
        yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
        thresh = hover_dist_thresh * max(xspan, yspan)
        if dist[nearest] < thresh:
            print("Clicked word:", words[nearest])

    # connect events
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    return {"pca": pca, "coords": X2, "words": words, "centroid": centroid}


# ---------------------------
# Example usage:
# ---------------------------
if __name__ == "__main__":
    # Example: random demo embeddings (replace with your real dict)
    demo_embeddings = {
        "apple": np.random.normal(size=64) + np.linspace(0, 1, 64),
        "banana": np.random.normal(size=64) + np.linspace(0.1, 0.9, 64),
        "car": np.random.normal(size=64) - np.linspace(0, 1, 64),
        "train": np.random.normal(size=64) - np.linspace(0.2, 1.2, 64),
        "cat": np.random.normal(size=64) + np.sin(np.linspace(0, 6, 64)),
        "dog": np.random.normal(size=64) + np.cos(np.linspace(0, 6, 64)),
    }

    # Replace demo_embeddings with your dict: my_dict = {"word": embedding_array, ...}
    result = plot_embeddings_pca(demo_embeddings, title="Demo: Word Embeddings PCA")

    # result["coords"] contains the 2D coordinates and result["centroid"] the centroid.
    print("Centroid (PC space):", result["centroid"])
    print("First few word coords:")
    for w, coord in zip(result["words"], result["coords"]):
        print(f"  {w}: ({coord[0]:.3f}, {coord[1]:.3f})")