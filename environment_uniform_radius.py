"""Kanami: Task2a 2D simulator. No ROS or GPU required for the smoke test."""
from pathlib import Path
from collections import deque
import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt


class Task2aCore:
    # Map-frame directions; robot yaw is fixed at +90 degrees.
    ACTIONS = np.array([(0, 1), (0, -1), (-1, 0), (1, 0),
                        (-1, 1), (1, 1), (-1, -1), (1, -1)], dtype=float)
    ACTIONS /= np.linalg.norm(ACTIONS, axis=1)[:, None]

    def __init__(self, seed=42):
        folder = Path(__file__).parent
        cfg = yaml.safe_load((folder / 'task2a_map.yaml').read_text())
        if abs(cfg['origin'][2]) > 1e-8 or cfg['negate'] != 0:
            raise ValueError('This initial version requires zero map yaw and negate=0')
        self.resolution = float(cfg['resolution'])
        self.origin = np.array(cfg['origin'][:2], dtype=float)
        self.image = np.array(Image.open(folder / cfg['image']))[::-1]
        self.free = (1 - self.image / 255.0) < cfg['free_thresh']
        # Unknown map cells are blocked. Subtract cell half-diagonal.
        self.clearance = (distance_transform_edt(self.free) * self.resolution
                          - self.resolution / np.sqrt(2))
        self.robot_radius = .30
        self.margin = .03  # 2次元環境の比較用。Gazebo用には再確認。originally .10
        self.obstacle_radius = .19
        self.step_size = .15
        self.start = np.array([2.50, 1.85])
        self.goal = np.array([2.00, 3.50])
        self.max_steps = 150
        self.rng = np.random.default_rng(seed)
        self.obstacles = np.empty((0, 2))
        self.detected = np.empty((0, 2))
        self.position = self.start.copy()
        self.steps = 0
        self.done = False

    def cells(self, xy):
        return np.floor((np.asarray(xy) - self.origin) / self.resolution).astype(int)

    def static_clearance(self, xy):
        x, y = self.cells(xy)
        if x < 0 or y < 0 or y >= len(self.free) or x >= self.free.shape[1]:
            return -1.0
        return self.clearance[y, x]

    def valid(self, xy):
        # Experimental ROI, not a surveyed physical boundary.
        if not (0.0 <= xy[0] <= 3.2 and 1.2 <= xy[1] <= 4.2):
            return False
        if self.static_clearance(xy) <= self.robot_radius + self.margin:
            return False
        return not len(self.obstacles) or np.all(
            np.linalg.norm(self.obstacles - xy, axis=1) >
            self.robot_radius + self.obstacle_radius + self.margin)

    def segment_valid(self, a, b):
        # Sample more densely than map cells, with conservative raster clearance.
        count = max(2, int(np.ceil(np.linalg.norm(b-a) / .01)) + 1)
        return all(self.valid(p) for p in np.linspace(a, b, count))

    def reachable(self):
        """Placement validation only: the DQN does not receive this search path."""
        spacing = .05
        origin = np.array([0.0, 1.2])
        shape = (65, 61)
        start = tuple(np.rint((self.start-origin)/spacing).astype(int))
        queue, visited = deque([start]), {start}
        while queue:
            cell = queue.popleft()
            point = origin + np.array(cell) * spacing
            if np.linalg.norm(point-self.goal) <= .10:
                return True
            for delta in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nxt = (cell[0]+delta[0], cell[1]+delta[1])
                if nxt in visited or not (0 <= nxt[0] < shape[0] and
                                          0 <= nxt[1] < shape[1]):
                    continue
                target = origin + np.array(nxt) * spacing
                if self.segment_valid(point, target):
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    def reset(self, seed=None, obstacle_count=0):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.position = self.start.copy()
        self.steps, self.done = 0, False
        self.detected = np.empty((0, 2))
        # Provisional placement region, based on observed Task2a coordinates.
        for _ in range(200):
            self.obstacles = self.rng.uniform([1.00, 2.85], [2.85, 3.80],
                                             size=(obstacle_count, 2))
            if any(self.static_clearance(p) <= self.obstacle_radius
                   for p in self.obstacles):
                continue
            if len(self.obstacles) > 1:
                distances = np.linalg.norm(self.obstacles[:, None] -
                                           self.obstacles[None, :], axis=2)
                np.fill_diagonal(distances, np.inf)
                if distances.min() <= 2*self.obstacle_radius:
                    continue
            if self.valid(self.start) and self.valid(self.goal) and self.reachable():
                return self.observation(), {'obstacles': self.obstacles.copy()}
        raise RuntimeError('Could not generate reachable placement; inspect geometry')

    def observation(self):
        # Approximate camera: +map-y facing, 100 deg FOV, 1.5 m range.
        # Static map occlusion; ideal detections in this initial version.
        for obstacle in self.obstacles:
            relative = obstacle-self.position
            if np.linalg.norm(relative) > 1.5 or relative[1] <= 0:
                continue
            if abs(np.arctan2(relative[0], relative[1])) > np.deg2rad(50):
                continue
            ray = np.linspace(self.position, obstacle,
                              max(2, int(np.linalg.norm(relative)/.025)+1))
            if any(self.static_clearance(p) < 0 for p in ray):
                continue
            if not len(self.detected) or np.min(
                    np.linalg.norm(self.detected-obstacle, axis=1)) > .01:
                self.detected = np.vstack([self.detected, obstacle])
        axis = (np.arange(30) - 14.5)*.10
        dx, dy = np.meshgrid(axis, axis)
        points = self.position + np.stack([dx, dy], axis=-1)
        static = np.array([self.static_clearance(p) <= self.robot_radius+self.margin
                           for p in points.reshape(-1, 2)]).reshape(30, 30)
        dynamic = np.zeros((30, 30), dtype=float)
        for obstacle in self.detected:
            dynamic = np.maximum(dynamic, np.linalg.norm(points-obstacle, axis=-1)
                                 <= self.robot_radius+self.obstacle_radius+self.margin)
        # Third channel: approximate observation coverage (not ground-truth occupancy).
        seen = ((dy > 0) & (np.hypot(dx, dy) <= 1.5) &
                (np.abs(np.arctan2(dx, dy)) <= np.deg2rad(50)))
        for row, col in zip(*np.where(seen)):
            ray = np.linspace(self.position, points[row, col], 30)
            if any(self.static_clearance(p) < 0 for p in ray):
                seen[row, col] = False
        return np.concatenate([static.ravel(), dynamic.ravel(), seen.ravel(),
                               np.clip((self.goal-self.position)/3., -1, 1)]).astype(np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError('Call reset() after episode ends')
        if not 0 <= int(action) < 8:
            raise ValueError('action must be 0..7')
        before = np.linalg.norm(self.goal-self.position)
        target = self.position + self.ACTIONS[int(action)]*self.step_size
        self.steps += 1
        collision = not self.segment_valid(self.position, target)
        if not collision:
            self.position = target
        after = np.linalg.norm(self.goal-self.position)
        success = not collision and after <= .12
        reward = -100. if collision else (100. if success else 5*(before-after)-.1)
        terminated = bool(collision or success)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        self.done = terminated or truncated
        return self.observation(), float(reward), terminated, truncated, {
            'success': success, 'collision_or_boundary': collision,
            'position': self.position.copy(), 'steps': self.steps}


if __name__ == '__main__':
    env = Task2aCore()
    obs, info = env.reset(seed=42)
    print('Observation shape:', obs.shape)
    print('Randomized obstacles:', info['obstacles'])
    print('Start/goal valid and connected:', env.valid(env.start),
          env.valid(env.goal), env.reachable())
    print('This is an environment check, NOT a trained DQN.')
