"""Plot one success/collision/timeout example, using each obstacle's radius."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from stable_baselines3 import DQN
from train_dqn import Task2aEnv

folder = Path(__file__).resolve().parent
env = Task2aEnv(obstacle_count=2)
model = DQN.load(str(folder / 'task2a_dqn_2_obstacles_6000steps.zip'), device='cpu')
examples = {}
for number, seed in enumerate(range(10000, 10100), 1):
    obs, info = env.reset(seed=seed)
    path = [env.core.position.copy()]
    obstacles = info['obstacles'].copy()
    radii = info['radii'].copy()
    labels = list(info['labels'])
    for _ in range(env.core.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        action = int(np.asarray(action).item())
        target = env.core.position + env.core.ACTIONS[action]*env.core.step_size
        obs, _, terminated, truncated, info = env.step(action)
        path.append(env.core.position.copy())
        if terminated or truncated:
            result = ('success' if info['success'] else
                      'collision_or_boundary' if info['collision_or_boundary'] else 'timeout')
            if result not in examples:
                examples[result] = (seed, np.array(path), obstacles, radii, labels,
                                    target.copy(), info['steps'])
                print('Found %s: seed=%d, steps=%d' % (result, seed, info['steps']), flush=True)
            break
    if number % 10 == 0:
        print('Checked %d/100 placements' % number, flush=True)
    if len(examples) == 3:
        break

core = env.core
h, w = core.image.shape
extent = (core.origin[0], core.origin[0]+w*core.resolution,
          core.origin[1], core.origin[1]+h*core.resolution)
fig, axes = plt.subplots(1, 3, figsize=(16, 7))
for ax, result in zip(axes, ('success', 'collision_or_boundary', 'timeout')):
    ax.imshow(core.image, origin='lower', extent=extent, cmap='gray', vmin=0, vmax=254)
    ax.scatter(*core.start, color='green', s=70, label='Start')
    ax.scatter(*core.goal, color='purple', marker='*', s=140, label='Goal')
    if result in examples:
        seed, path, obstacles, radii, labels, target, steps = examples[result]
        for p, radius, label in zip(obstacles, radii, labels):
            ax.add_patch(Circle(p, radius, color='red', alpha=.45))
            ax.add_patch(Circle(p, core.robot_radius+radius+core.margin,
                               fill=False, color='red', linestyle='--'))
            ax.annotate('%s\nr=%.2f m' % (label, radius), p,
                        xytext=(5, 8), textcoords='offset points', fontsize=8)
        ax.plot(path[:, 0], path[:, 1], 'b.-', label='Robot-center path')
        ax.scatter(*path[-1], marker='x', color='black', s=80, label='Final position')
        if result == 'collision_or_boundary':
            ax.plot([path[-1, 0], target[0]], [path[-1, 1], target[1]],
                    color='orange', linestyle='--', linewidth=3, label='Rejected move')
        ax.set_title('%s\nseed=%d, steps=%d' % (result, seed, steps))
    else:
        ax.set_title(result + ': no example found')
    ax.set(xlim=(-.1, 3.4), ylim=(1., 4.4), xlabel='map x [m]', ylabel='map y [m]')
    ax.set_aspect('equal')
    ax.grid(alpha=.2)
    ax.legend(fontsize=7)
fig.suptitle('Individual obstacle radii; robot=%.2f m, margin=%.2f m' %
             (core.robot_radius, core.margin))
fig.tight_layout()
number = 1
while (folder / ('dqn_paths_%d.png' % number)).exists():
    number += 1
output = folder / ('dqn_paths_%d.png' % number)
fig.savefig(output, dpi=180)
for result, example in examples.items():
    path = example[1]
    np.savetxt(folder / ('trajectory_%s_%d.csv' % (result, number)),
               np.column_stack([np.arange(len(path)), path]), delimiter=',',
               header='step,x,y', comments='')
plt.close(fig)
print('Saved:', output, flush=True)
