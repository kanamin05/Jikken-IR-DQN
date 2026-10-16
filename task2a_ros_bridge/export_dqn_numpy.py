"""Export the existing SB3 MLP to a Python-3.6-compatible NumPy archive."""
import argparse
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import DQN
from environment import Task2aCore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='task2a_dqn_2_obstacles_6000steps.zip')
    parser.add_argument('--output', default='task2a_policy_numpy.npz')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Output already exists; choose another --output')
    model = DQN.load(args.model, device='cpu')
    if model.observation_space.shape != (2702,) or model.action_space.n != 8:
        raise ValueError('Expected 2702 observations and 8 actions')
    if model.q_net.features_extractor.__class__.__name__ != 'FlattenExtractor':
        raise ValueError('Only the original FlattenExtractor MLP is supported')
    layers = list(model.q_net.q_net.children())
    for i, layer in enumerate(layers):
        expected = torch.nn.Linear if i % 2 == 0 else torch.nn.ReLU
        if not isinstance(layer, expected):
            raise ValueError('Only alternating Linear/ReLU ending in Linear is supported')
    if len(layers) % 2 != 1:
        raise ValueError('Network must end in Linear')
    linear = [layer for layer in layers if isinstance(layer, torch.nn.Linear)]
    core = Task2aCore()
    data = {'n_layers': np.array(len(linear)),
            'clearance': core.clearance, 'free': core.free,
            'resolution': np.array(core.resolution), 'origin': core.origin,
            'goal': core.goal, 'start': core.start, 'actions': core.ACTIONS,
            'robot_radius': np.array(core.robot_radius), 'margin': np.array(core.margin),
            'step_size': np.array(core.step_size),
            'labels': np.array(list(core.OBSTACLE_RADII)),
            'radii': np.array(list(core.OBSTACLE_RADII.values())),
            'version': np.array(1)}
    for i, layer in enumerate(linear):
        data['w%d' % i] = layer.weight.detach().cpu().numpy().copy()
        data['b%d' % i] = layer.bias.detach().cpu().numpy().copy()
    rng = np.random.default_rng(42)
    samples = [rng.uniform(-1, 1, 2702).astype(np.float32) for _ in range(16)]
    for seed in range(3):
        obs, _ = core.reset(seed=seed, obstacle_count=2)
        samples.append(obs)
    samples = np.stack(samples)
    x = samples.copy()
    for i in range(len(linear)):
        x = x @ data['w%d' % i].T + data['b%d' % i]
        if i < len(linear)-1:
            x = np.maximum(x, 0)
    with torch.no_grad():
        reference = model.q_net(torch.as_tensor(samples)).cpu().numpy()
    np.testing.assert_allclose(x, reference, rtol=1e-4, atol=1e-4)
    if not np.array_equal(x.argmax(axis=1), reference.argmax(axis=1)):
        raise ValueError('NumPy and SB3 actions differ')
    np.savez_compressed(output, **data)
    print('Exported:', output.resolve())
    print('SB3/NumPy action agreement: %d/%d' % (len(samples), len(samples)))
    print('Max Q error:', float(np.abs(x-reference).max()))


if __name__ == '__main__':
    main()
