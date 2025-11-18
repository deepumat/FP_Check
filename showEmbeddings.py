"""
PCA scatter for two sets of word embeddings (main + extra).

Features:
 - Projects both sets with the same PCA.
 - Main and extra points plotted in different colors/markers/sizes.
 - Centroid of the main set is shown.
 - Hovering shows the word label.
 - Clicking on a point prints the word.
 - Optional arrows drawn from centroid to extra points.
 - Returns PCA, coords, words, centroid for programmatic use.

Requirements:
    pip install numpy matplotlib scikit-learn
Run in an environment with an interactive matplotlib backend (Jupyter, or python with TkAgg/Qt5Agg).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Dict, Tuple, Optional

def plot_two_embedding_sets(
    main_embeddings: Dict[str, np.ndarray],
    extra_embeddings: Optional[Dict[str, np.ndarray]] = None,
    title: str = "PCA of Embeddings",
    main_color: str = "tab:blue",
    extra_color: str = "tab:orange",
    main_marker: str = "o",
    extra_marker: str = "D",
    main_marker_size: int = 60,
    extra_marker_size: int = 80,
    centroid_color: str = "black",
    centroid_marker: str = "X",
    centroid_marker_size: int = 220,
    hover_dist_thresh: float = 0.04,
    draw_arrows_to_extra: bool = False,
    show_grid: bool = True,
    show_legend: bool = True
) -> Dict:
    """
    Plot two embedding sets after projecting with the same PCA.
    Args:
        main_embeddings: dict[word] -> 1D array-like
        extra_embeddings: optional dict[word] -> 1D array-like
        draw_arrows_to_extra: if True, draw arrows from centroid to each extra point
    Returns:
        dict with keys: pca, main_coords, extra_coords, centroid, words_main, words_extra
    """

    # --- Input checks
    if not isinstance(main_embeddings, dict) or len(main_embeddings) == 0:
        raise ValueError("main_embeddings must be a non-empty dict {word: vector}")

    words_main = list(main_embeddings.keys())
    X_main = np.vstack([np.asarray(main_embeddings[w]) for w in words_main])
    if X_main.ndim != 2:
        raise ValueError("Main embeddings should be 1D vectors of equal length")

    if extra_embeddings:
        words_extra = list(extra_embeddings.keys())
        X_extra = np.vstack([np.asarray(extra_embeddings[w]) for w in words_extra])
        if X_extra.shape[1] != X_main.shape[1]:
            raise ValueError("Main and extra embeddings must have same dimensionality")
        X_all = np.vstack([X_main, X_extra])
        all_words = words_main + words_extra
    else:
        words_extra = []
        X_extra = np.zeros((0, X_main.shape[1]))
        X_all = X_main
        all_words = words_main

    # --- PCA to 2D
    pca = PCA(n_components=2)
    X2_all = pca.fit_transform(X_all)
    X2_main = X2_all[:len(words_main)]
    X2_extra = X2_all[len(words_main):] if len(words_extra) else np.zeros((0,2))
    centroid = X2_main.mean(axis=0)

    # --- Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    pts_main = ax.scatter(
        X2_main[:,0], X2_main[:,1],
        s=main_marker_size,
        c=main_color,
        marker=main_marker,
        label="Main set",
        edgecolors="w",
        linewidths=0.6,
        zorder=2
    )

    pts_extra = None
    if len(words_extra):
        pts_extra = ax.scatter(
            X2_extra[:,0], X2_extra[:,1],
            s=extra_marker_size,
            c=extra_color,
            marker=extra_marker,
            label="Extra set",
            edgecolors="w",
            linewidths=0.6,
            zorder=3
        )

    # centroid marker
    ax.scatter([centroid[0]], [centroid[1]],
               s=centroid_marker_size,
               c=centroid_color,
               marker=centroid_marker,
               label="Centroid (main)",
               zorder=4)

    # Optional arrows from centroid -> each extra point
    if draw_arrows_to_extra and len(words_extra):
        for ex_pt in X2_extra:
            ax.annotate(
                "", xy=(ex_pt[0], ex_pt[1]), xytext=(centroid[0], centroid[1]),
                arrowprops=dict(arrowstyle="->", color=extra_color, alpha=0.6, linewidth=1.2),
                zorder=1
            )

    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    if show_grid:
        ax.grid(True, linestyle="--", alpha=0.35)
    if show_legend:
        ax.legend()

    # --- Hover annotation
    annot = ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    # Combined coords and words for easy nearest lookup
    coords = X2_all
    words_all = all_words

    def update_annot(i:int):
        annot.xy = (coords[i,0], coords[i,1])
        annot.set_text(words_all[i])
        annot.get_bbox_patch().set_alpha(0.95)

    def on_move(event):
        if event.inaxes != ax:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return

        # compute distances
        dx = coords[:,0] - event.xdata
        dy = coords[:,1] - event.ydata
        dist = np.hypot(dx, dy)

        # threshold in data units
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

    # Clicking prints the word and which set it belongs to
    def on_click(event):
        if event.inaxes != ax:
            return
        dx = coords[:,0] - event.xdata
        dy = coords[:,1] - event.ydata
        dist = np.hypot(dx, dy)
        xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
        yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
        thresh = hover_dist_thresh * max(xspan, yspan)
        nearest = np.argmin(dist)
        if dist[nearest] < thresh:
            word = words_all[nearest]
            set_name = "main" if nearest < len(words_main) else "extra"
            print(f"Clicked: '{word}'  (set: {set_name})")

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.tight_layout()
    plt.show()

    return {
        "pca": pca,
        "main_coords": X2_main,
        "extra_coords": X2_extra,
        "centroid": centroid,
        "words_main": words_main,
        "words_extra": words_extra
    }


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Main embeddings (example)
    main_emb = {
        "apple": np.random.normal(size=128) + np.linspace(0,1,128),
        "banana": np.random.normal(size=128) + np.linspace(0.1,0.9,128),
        "cat": np.random.normal(size=128) + np.sin(np.linspace(0,6,128)),
        "dog": np.random.normal(size=128) + np.cos(np.linspace(0,6,128)),
    }

    # Extra embeddings (example)
    extra_emb = {
        "orange": np.random.normal(size=128),
        "lion": np.random.normal(size=128),
        "car": np.random.normal(size=128),
    }

    # Call with arrows, different marker/size, and interactive hover/click enabled
    result = plot_two_embedding_sets(
        main_emb,
        extra_emb,
        title="Main vs Extra Embeddings (PCA)",
        main_color="tab:blue",
        extra_color="tab:orange",
        main_marker="o",
        extra_marker="s",
        main_marker_size=80,
        extra_marker_size=120,
        draw_arrows_to_extra=True
    )

    print("Centroid (PC space):", result["centroid"])
    print("Main coords shape:", result["main_coords"].shape)
    print("Extra coords shape:", result["extra_coords"].shape)