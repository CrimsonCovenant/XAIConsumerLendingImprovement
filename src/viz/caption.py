"""Companion-caption writer for saved figures.
Every saved figure can use save_with_caption() to write a companion
.caption.md file alongside the PNG. This gives non-technical readers
a plain-English explanation of what the chart shows.
"""
import os
import matplotlib.pyplot as plt

def save_with_caption(fig, png_path: str, caption_md: str) -> None:
    """Save a matplotlib figure and write its companion caption file.
    fig: the figure to save.
    png_path: destination path for the PNG.
    caption_md: 2-3 sentence plain-English explanation of the chart.
    """
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path)
    plt.close(fig)
    cap_path = png_path.rsplit(".", 1)[0] + ".caption.md"
    with open(cap_path, "w") as f:
        f.write(caption_md.strip() + "\n")
