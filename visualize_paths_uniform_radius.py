from pathlib import Path
import numpy as np
import matplotlib

# コンテナ内で画面がなくてもPNGを保存できるようにする
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from stable_baselines3 import DQN
from train_dqn import Task2aEnv


folder = Path(__file__).resolve().parent
model = DQN.load(str(folder / "task2a_dqn.zip"), device="cpu")
env = Task2aEnv(obstacle_count=1)

# 結果の種類ごとに1例を保存する
examples = {}

for seed in range(10000, 10100):
    obs, _ = env.reset(seed=seed)
    obstacles = env.core.obstacles.copy()
    trajectory = [env.core.position.copy()]
    attempted_target = None

    for _ in range(env.core.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        action = int(np.asarray(action).item())

        # 衝突時は環境が位置を更新しないため、
        # 実行しようとした移動先も記録する
        attempted_target = (
            env.core.position
            + env.core.ACTIONS[action] * env.core.step_size
        )

        obs, _, terminated, truncated, info = env.step(action)
        trajectory.append(env.core.position.copy())

        if terminated or truncated:
            if info["success"]:
                result = "success"
            elif info["collision_or_boundary"]:
                result = "collision_or_boundary"
            else:
                result = "timeout"

            if result not in examples:
                examples[result] = {
                    "seed": seed,
                    "trajectory": np.array(trajectory),
                    "obstacles": obstacles,
                    "attempted_target": attempted_target.copy(),
                    "steps": info["steps"],
                }
                print(
                    "Found %s: seed=%d, steps=%d"
                    % (result, seed, info["steps"]),
                    flush=True
                )
            break

    if len(examples) == 3:
        break


core = env.core
height, width = core.image.shape
x0, y0 = core.origin
extent = (
    x0, x0 + width * core.resolution,
    y0, y0 + height * core.resolution
)

fig, axes = plt.subplots(1, 3, figsize=(16, 7))
results = ("success", "collision_or_boundary", "timeout")

for ax, result in zip(axes, results):
    # 黒=壁、白=空き、灰色=未知
    ax.imshow(
        core.image, origin="lower", extent=extent,
        cmap="gray", vmin=0, vmax=254
    )

    # ロボット中心の移動を制限している評価領域
    ax.plot(
        [0, 3.2, 3.2, 0, 0],
        [1.2, 1.2, 4.2, 4.2, 1.2],
        color="gray", linestyle=":", label="ROI boundary"
    )

    ax.scatter(
        *core.start, color="green", marker="o",
        s=70, label="Start", zorder=5
    )
    ax.scatter(
        *core.goal, color="purple", marker="*",
        s=140, label="Goal", zorder=5
    )

    if result not in examples:
        ax.set_title(result + ": no example found")
    else:
        example = examples[result]
        path = example["trajectory"]

        for index, obstacle in enumerate(example["obstacles"]):
            # 赤い円=障害物そのものの近似
            ax.add_patch(Circle(
                obstacle, core.obstacle_radius,
                color="red", alpha=0.45,
                label="Obstacle" if index == 0 else None
            ))

            # 赤い破線=ロボット中心が入れない範囲
            ax.add_patch(Circle(
                obstacle,
                core.robot_radius + core.obstacle_radius + core.margin,
                fill=False, edgecolor="red", linestyle="--",
                label="Exclusion radius" if index == 0 else None
            ))

        ax.plot(
            path[:, 0], path[:, 1],
            color="blue", marker=".", markersize=4,
            linewidth=1.5, label="Robot-center path"
        )
        ax.scatter(
            *path[-1], color="black", marker="x",
            s=80, label="Final position", zorder=6
        )

        # 衝突・範囲外の場合、拒否された移動を描く
        if result == "collision_or_boundary":
            target = example["attempted_target"]
            ax.plot(
                [path[-1, 0], target[0]],
                [path[-1, 1], target[1]],
                color="orange", linestyle="--", linewidth=3,
                label="Rejected move"
            )

        ax.set_title(
            "%s\nseed=%d, steps=%d"
            % (result, example["seed"], example["steps"])
        )

        # 往復や停止はPNGだと重なるので、数値でも保存する
        np.savetxt(
            folder / ("trajectory_%s.csv" % result),
            np.column_stack([np.arange(len(path)), path]),
            delimiter=",", header="step,x,y", comments=""
        )

    ax.set_xlim(-0.1, 3.4)
    ax.set_ylim(1.0, 4.4)
    ax.set_aspect("equal")
    ax.set_xlabel("map x [m]")
    ax.set_ylabel("map y [m]")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="upper left")

fig.tight_layout()
# 既存ファイルを上書きせず、番号を増やして保存
number = 1
while (folder / ("dqn_paths_%d.png" % number)).exists():
    number += 1

output = folder / ("dqn_paths_%d.png" % number)

fig.savefig(output, dpi=180)
plt.close(fig)
print("Saved:", output)
