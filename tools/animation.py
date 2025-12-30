"""
npy2mp4
matplotlib==3.3.3
"""

import os
# 确保非交互后端，避免服务器/服务进程里无显示导致报错
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (matplotlib 3.3 仍需显式导入)
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from tqdm import tqdm


def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4):
    def init(fig, ax):
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        if title:
            fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xz_plane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz],
        ]
        xz_plane = Poly3DCollection([verts], alpha=0.5)
        xz_plane.set_facecolor((0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    data = joints.copy().reshape(len(joints), -1, 3)  # (seq_len, joints_num, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init(fig, ax)
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = [
        "red", "blue", "black", "red", "blue",
        "darkblue", "darkblue", "darkblue", "darkblue", "darkblue",
        "darkred", "darkred", "darkred", "darkred", "darkred",
    ]
    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        for artist in list(ax.lines):
            artist.remove()
        for artist in list(ax.collections):
            artist.remove()

        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xz_plane(
            MINS[0] - trajec[index, 0],
            MAXS[0] - trajec[index, 0],
            0,
            MINS[2] - trajec[index, 1],
            MAXS[2] - trajec[index, 1],
        )

        if index > 1:
            ax.plot3D(
                trajec[:index, 0] - trajec[index, 0],
                np.zeros_like(trajec[:index, 0]),
                trajec[:index, 1] - trajec[index, 1],
                linewidth=1.0,
                color="blue",
            )

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            linewidth = 4.0 if i < 5 else 2.0
            ax.plot3D(
                data[index, chain, 0],
                data[index, chain, 1],
                data[index, chain, 2],
                linewidth=linewidth,
                color=color,
            )

        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()


def npy2mp4(npy_folder, mp4_folder, kinematic_chain, fps=30):
    # 只处理 *_out.npy，避免把*_in.npy转换
    npy_files = []
    for root, _, files in os.walk(npy_folder):
        for file in files:
            if file.endswith("_out.npy"):
                npy_files.append(os.path.join(root, file))
    npy_files.sort()

    total = len(npy_files)
    if total == 0:
        print("NO_FILES")
        return

    os.makedirs(mp4_folder, exist_ok=True)

    for i, npy_file in enumerate(tqdm(npy_files, desc="npy2mp4")):
        try:
            data = np.load(npy_file, allow_pickle=True).reshape(-1, 22, 3)
            assert data.shape[-2:] == (22, 3), f"Unexpected data shape for file: {npy_file}"

            npy_filename = os.path.splitext(os.path.basename(npy_file))[0]
            save_path = os.path.join(mp4_folder, f"{npy_filename}.mp4")

            # 已存在就跳过（便于断点继续）
            if os.path.exists(save_path):
                print(f"FILE_DONE {os.path.basename(save_path)}")  # 让后端能统计进度
                print(f"PROGRESS {min(i + 1, total)}/{total}")
                continue

            plot_3d_motion(save_path, kinematic_chain, data, title=None, fps=fps, radius=4)

            print(f"FILE_DONE {os.path.basename(save_path)}")
            print(f"PROGRESS {i + 1}/{total}")

        except Exception as e:
            # 单个失败不终止整体流程
            print(f"FILE_ERROR {os.path.basename(npy_file)} {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy-folder", required=True, help="包含 *_out.npy 的 samples 目录")
    parser.add_argument("--mp4-folder", default=None, help="输出 mp4 的目录（默认 <npy-folder>/animation）")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--ffmpeg", default=os.getenv("FFMPEG_BIN"), help="ffmpeg 可执行文件路径（可用环境变量 FFMPEG_BIN）")
    args = parser.parse_args()

    npy_folder = args.npy_folder
    mp4_folder = args.mp4_folder or os.path.join(npy_folder, "animation")

    # 可选：指定 ffmpeg 的路径
    if args.ffmpeg and os.path.exists(args.ffmpeg):
        plt.rcParams["animation.ffmpeg_path"] = args.ffmpeg

    # 你原来的骨架链（人体拓扑）
    kinematic_chain = [
        [0, 2, 5, 8, 11],
        [0, 1, 4, 7, 10],
        [0, 3, 6, 9, 12, 15],
        [9, 14, 17, 19, 21],
        [9, 13, 16, 18, 20],
    ]

    print(f"NPY_DIR={npy_folder}")
    print(f"MP4_DIR={mp4_folder}")

    npy2mp4(npy_folder, mp4_folder, kinematic_chain, fps=args.fps)

    print("ALL_DONE")


if __name__ == "__main__":
    main()
