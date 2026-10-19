import unittest
import numpy as np
from scipy.spatial import cKDTree
from task2a_precise_run import SurfaceDQN
from task2a_fixed_run import forward_waypoints, segment_report


class FixedRouteTests(unittest.TestCase):
    def policy(self):
        p = SurfaceDQN.__new__(SurfaceDQN)
        p.tree = cKDTree([[1.5, 2.248]])
        p.body = .33
        p.sample = lambda grid, points, outside: np.ones(len(points))
        p.data = {'clearance': None}
        return p

    def test_surface_margin_is_not_relaxed_at_entry(self):
        p = self.policy(); start = np.array([1.5, 2.]); target = start+[0., -.04]
        self.assertEqual(segment_report(p, start, target)['rejected_by'], ['surface'])
        self.assertEqual(segment_report(p, start, start+[0., .04])['rejected_by'], ['surface'])

    def test_forward_first_uses_actual_heading_without_sideways_entry(self):
        position = np.array([2.65, 2.49])
        heading = np.deg2rad(80.)
        points = forward_waypoints(position, heading, np.array([2., 3.5]))
        np.testing.assert_allclose(points[0], [2.65+1.01/np.tan(heading), 3.5])
        np.testing.assert_allclose(points[1], [2., 3.5])
        north = forward_waypoints(position, np.pi/2, np.array([2., 3.5]))
        np.testing.assert_allclose(north[0], [2.65, 3.5])

    def test_forward_refuses_goal_behind_or_parallel_heading(self):
        for heading in [0., -np.pi/2]:
            with self.assertRaises(ValueError):
                forward_waypoints(np.array([2.65, 2.49]), heading, np.array([2., 3.5]))


if __name__ == '__main__':
    unittest.main()
