from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

LABEL_FONTSIZE = 12
FPS = 30
DPI = 100


def animate_stimulus(
    video: np.ndarray,
    filename: Path = None,
):
    h, w = video.shape[1], video.shape[2]
    f_h, f_w = (h / 16) + (7 / h), w / 16
    a_w = 0.99
    a_h = (h / 16) / f_h

    figure = plt.figure(figsize=(f_w, f_h), dpi=DPI, facecolor="white")
    ax = figure.add_axes(rect=((1 - a_w) / 2, (1 - a_w) / 2, a_w, a_h))

    imshow = ax.imshow(
        np.random.rand(h, w),
        cmap="gray",
        aspect="equal",
        vmin=0,
        vmax=1,
    )
    pos = ax.get_position()
    text = ax.text(
        x=0,
        y=pos.y1 + 0.11,
        s="",
        ha="left",
        va="center",
        fontsize=LABEL_FONTSIZE,
        transform=ax.transAxes,
    )
    ax.grid(linewidth=0)
    ax.set_xticks([])
    ax.set_yticks([])

    def animate(frame: int):
        imshow.set_data(video[frame, :, :])
        text.set_text(f"Movie Frame {frame:03d}")

    anim = FuncAnimation(
        figure, animate, frames=video.shape[0], interval=int(1000 / FPS)
    )
    if filename is not None:
        filename.parent.mkdir(parents=True, exist_ok=True)
        anim.save(filename, fps=FPS, dpi=DPI, savefig_kwargs={"pad_inches": 0})
        try:
            plt.close(figure)
        except AttributeError as e:
            print(f"AttributeError in plt.close(figure): {e}.")
    plt.close(figure)


def main():
    output_dir = Path("runs") / "VT333_FOV1_day1"
    video = np.load(output_dir / "video" / "zebra_noise_downsampled.npy")
    video = video.astype(np.float32)

    animate_stimulus(video=video, filename=output_dir / "video" / "downsampled.mp4")


if __name__ == "__main__":
    main()
