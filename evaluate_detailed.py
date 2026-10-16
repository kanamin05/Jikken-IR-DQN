"""Kanami: evaluate the saved Task2a DQN and classify rejected moves.

Put next to environment.py and train_dqn.py. Does not train or change models.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def point_reasons(core, point):
    """Mirror current Task2aCore.valid; thresholds denote clearance violations."""
    reasons, hits = [], []
    if not (0 <= point[0] <= 3.2 and 1.2 <= point[1] <= 4.2):
        reasons.append('roi')
    if core.static_clearance(point) <= core.robot_radius + core.margin:
        reasons.append('static_clearance')
    for index, (obstacle, radius) in enumerate(zip(core.obstacles, core.obstacle_radii)):
        if np.linalg.norm(point-obstacle) <= core.robot_radius+radius+core.margin:
            hits.append(index)
    if hits:
        reasons.append('obstacle_clearance')
    return reasons, hits


def rejected_segment(core, start, target, known_before):
    # Same sampling rule and order as Task2aCore.segment_valid.
    count = max(2, int(np.ceil(np.linalg.norm(target-start)/.01))+1)
    for point in np.linspace(start, target, count):
        if not core.valid(point):
            reasons, hits = point_reasons(core, point)
            category = reasons[0] if len(reasons) == 1 else ('mixed' if reasons else 'unclassified')
            return {
                'category': category,
                'reasons': reasons,
                'first_invalid_point': point.copy(),
                'obstacle_hits': [
                    {'index': i, 'label': core.obstacle_labels[i],
                     'known_before_action': i in known_before}
                    for i in hits
                ],
            }
    return {'category': 'unclassified', 'reasons': [], 'obstacle_hits': []}


def rollout(env, model, seed):
    obs, _ = env.reset(seed=seed)
    core = env.core
    record = {
        'seed': seed,
        'obstacles': core.obstacles.copy(),
        'labels': list(core.obstacle_labels),
        'radii': core.obstacle_radii.copy(),
        'path': [core.position.copy()],
        'actions': [], 'known_before_actions': [],
        'return': 0., 'diagnostic': None,
    }
    for step in range(1, core.max_steps+1):
        action, _ = model.predict(obs, deterministic=True)
        action = int(np.asarray(action).item())
        start = core.position.copy()
        known_before = set(core.detected_indices)
        target = start + core.ACTIONS[action]*core.step_size
        record['actions'].append(action)
        record['known_before_actions'].append(sorted(known_before))
        obs, reward, terminated, truncated, info = env.step(action)
        record['return'] += float(reward)
        record['path'].append(core.position.copy())
        if terminated or truncated:
            if info['success']:
                outcome = 'success'
            elif info['collision_or_boundary']:
                record['diagnostic'] = rejected_segment(core, start, target, known_before)
                outcome = record['diagnostic']['category']
            else:
                outcome = 'timeout'
            record.update(outcome=outcome, steps=step, attempted_target=target.copy())
            record['path'] = np.array(record['path'])
            # Revisit statistic is descriptive, not a proof of a control loop.
            unique = len(np.unique(np.round(record['path'], 3), axis=0))
            record['unique_positions'] = unique
            record['revisit_fraction'] = 1-unique/len(record['path'])
            return record
    raise RuntimeError('Episode did not terminate/truncate within max_steps')


def save_figure(core, records, output, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    categories = ('success', 'obstacle_clearance', 'static_clearance',
                  'roi', 'mixed', 'timeout')
    counts = Counter(r['outcome'] for r in records)
    h, w = core.image.shape
    extent = (core.origin[0], core.origin[0]+w*core.resolution,
              core.origin[1], core.origin[1]+h*core.resolution)
    fig, axes = plt.subplots(2, 3, figsize=(15, 12))
    for ax, category in zip(axes.ravel(), categories):
        ax.imshow(core.image, origin='lower', extent=extent, cmap='gray', vmin=0, vmax=254)
        ax.plot([0, 3.2, 3.2, 0, 0], [1.2, 1.2, 4.2, 4.2, 1.2],
                ':', color='gray', label='ROI boundary')
        ax.scatter(*core.start, color='green', s=50, label='Start')
        ax.scatter(*core.goal, color='purple', marker='*', s=110, label='Goal')
        record = next((r for r in records if r['outcome'] == category), None)
        if record:
            for p, radius, label in zip(record['obstacles'], record['radii'], record['labels']):
                ax.add_patch(Circle(p, radius, color='red', alpha=.35))
                ax.add_patch(Circle(p, core.robot_radius+radius+core.margin,
                                   fill=False, color='red', linestyle='--'))
                ax.annotate(label, p, xytext=(3, 5), textcoords='offset points', fontsize=7)
            path = record['path']
            ax.plot(path[:, 0], path[:, 1], 'b.-', markersize=3, label='Robot-center path')
            ax.scatter(*path[-1], color='black', marker='x', s=70, label='Final')
            if record['diagnostic']:
                target = record['attempted_target']
                ax.plot([path[-1, 0], target[0]], [path[-1, 1], target[1]],
                        '--', color='orange', linewidth=3, label='Rejected move')
            subtitle = 'seed=%d, steps=%d' % (record['seed'], record['steps'])
        else:
            subtitle = 'No example in evaluated placements'
        ax.set_title('%s: %d\n%s' % (category, counts[category], subtitle))
        ax.set(xlim=(-.1, 3.4), ylim=(1., 4.4), xlabel='map x [m]', ylabel='map y [m]')
        ax.set_aspect('equal')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, loc='upper left')
    fig.suptitle(title+'\nClearance violations include the configured safety margin.')
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='task2a_dqn_2_obstacles_6000steps.zip')
    parser.add_argument('--obstacles', type=int, default=2, choices=(0, 1, 2))
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed-start', type=int, default=10000)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error('--episodes must be positive')
    folder = Path(__file__).resolve().parent
    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = folder/model_path
    if not model_path.is_file():
        parser.error('Model not found: '+str(model_path))
    from stable_baselines3 import DQN
    from train_dqn import Task2aEnv
    env = Task2aEnv(obstacle_count=args.obstacles)
    model = DQN.load(str(model_path), device='cpu')
    core = env.core
    for name in ('obstacle_radii', 'detected_indices', 'obstacle_labels'):
        if not hasattr(core, name):
            raise RuntimeError('Use the individual-size environment.py: missing '+name)
    number = 1
    while True:
        output = folder/('evaluation_%d' % number)
        try:
            output.mkdir()
            break
        except FileExistsError:
            number += 1
    meta = {
        'model': str(model_path), 'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'obstacle_count': args.obstacles, 'episodes': args.episodes,
        'seed_start': args.seed_start, 'robot_radius': core.robot_radius,
        'margin': core.margin, 'step_size': core.step_size,
        'max_steps': core.max_steps, 'radii_by_label': core.OBSTACLE_RADII,
        'environment_sha256': hashlib.sha256((folder/'environment.py').read_bytes()).hexdigest(),
        'map_yaml_sha256': hashlib.sha256((folder/'task2a_map.yaml').read_bytes()).hexdigest(),
        'map_pgm_sha256': hashlib.sha256((folder/'task2a_map.pgm').read_bytes()).hexdigest(),
        'note': 'Rejected-clearance violations are not necessarily physical contacts. '
                'Classification mirrors ROI and radius rules of the supplied environment.',
    }
    (output/'metadata.json').write_text(json.dumps(meta, indent=2, default=json_default))
    print('Model:', model_path, flush=True)
    print('Results folder:', output, flush=True)
    print('Robot radius=%.3f margin=%.3f' % (core.robot_radius, core.margin), flush=True)
    records, counts = [], Counter()
    with (output/'episodes.jsonl').open('w') as detail, (output/'results.csv').open('w', newline='') as table:
        writer = csv.writer(table)
        writer.writerow(['seed', 'outcome', 'steps', 'return', 'unique_positions', 'revisit_fraction'])
        for index, seed in enumerate(range(args.seed_start, args.seed_start+args.episodes), 1):
            record = rollout(env, model, seed)
            records.append(record)
            counts[record['outcome']] += 1
            detail.write(json.dumps(record, default=json_default)+'\n')
            detail.flush()
            writer.writerow([record[k] for k in ('seed', 'outcome', 'steps', 'return',
                                                 'unique_positions', 'revisit_fraction')])
            table.flush()
            if index % 10 == 0 or index == args.episodes:
                rejected = sum(v for k, v in counts.items() if k not in ('success', 'timeout'))
                print('Evaluated %d/%d: success=%d collision_or_boundary=%d timeout=%d' %
                      (index, args.episodes, counts['success'], rejected, counts['timeout']), flush=True)
    # Count per rejected episode; known/unseen counts can overlap when two obstacles are hit.
    obstacle_rejections = [r for r in records if r['diagnostic'] and r['diagnostic']['obstacle_hits']]
    summary = dict(meta)
    summary.update(counts=dict(counts), success_rate=counts['success']/args.episodes,
                   obstacle_rejection_episodes=len(obstacle_rejections),
                   hit_obstacle_known_before=sum(any(h['known_before_action'] for h in
                       r['diagnostic']['obstacle_hits']) for r in obstacle_rejections),
                   hit_obstacle_unseen_before=sum(any(not h['known_before_action'] for h in
                       r['diagnostic']['obstacle_hits']) for r in obstacle_rejections))
    lines = ['Success rate: %.1f%%' % (100*summary['success_rate'])]
    for category in ('success', 'obstacle_clearance', 'static_clearance', 'roi',
                     'mixed', 'unclassified', 'timeout'):
        lines.append('%s: %d/%d' % (category, counts[category], args.episodes))
    lines += ['Obstacle rejection episodes (including mixed): %d' % len(obstacle_rejections),
              '  Hit obstacle known before action: %d' % summary['hit_obstacle_known_before'],
              '  Hit obstacle unseen before action: %d' % summary['hit_obstacle_unseen_before']]
    (output/'summary.json').write_text(json.dumps(summary, indent=2, default=json_default))
    (output/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)
    save_figure(core, records, output/'paths.png',
                '%s | %d obstacles | success %.1f%%' %
                (model_path.name, args.obstacles, 100*summary['success_rate']))
    print('Saved:', output/'paths.png', flush=True)


if __name__ == '__main__':
    main()
