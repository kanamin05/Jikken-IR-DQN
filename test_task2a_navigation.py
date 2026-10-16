import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial import cKDTree
from task2a_cloud_infer import CloudDQN
from task2a_route_guard import RouteGuard
from task2a_precise_run import PreciseDQN, clear_observed_floor
from task2a_cloud_step import run_steps, velocity_command
from task2a_handoff import handoff_reason


def policy(cls=CloudDQN):
    p = cls.__new__(cls)
    p.shape, p.resolution, p.origin, p.body = (61, 65), .05, np.array([0., 1.2]), .05
    actions = np.array([(0, 1), (0, -1), (-1, 0), (1, 0),
                        (-1, 1), (1, 1), (-1, -1), (1, -1)], dtype=float)
    actions /= np.linalg.norm(actions, axis=1)[:, None]
    p.data = dict(clearance=np.ones(p.shape), goal=np.array([1.5, 3.5]),
                  actions=actions, step_size=.15)
    p.blocked = np.zeros(p.shape, dtype=bool)
    return p


class NavigationTests(unittest.TestCase):
    def test_guarded_finish_checks_surfaces_and_requires_route(self):
        p = policy(PreciseDQN)
        start = np.array([1.5, 2.])
        path = np.array([start, [1.5, 2.025], [1.5, 2.05]])
        p.tree = cKDTree([[3., 4.]])
        with patch.object(p, 'qvalues', side_effect=AssertionError('DQN called')):
            target, info = p.choose_guarded_route(start, path)
        self.assertEqual(info['mode'], 'guarded_route_finish')
        self.assertIsNone(info['action'])
        self.assertLessEqual(np.linalg.norm(target-start), .05+1e-9)
        p.tree = cKDTree([[1.5, 2.26]])
        with self.assertRaises(ValueError):
            p.choose_guarded_route(start, path)
        with self.assertRaises(ValueError):
            p.choose_guarded_route(start, None)

    def test_handoff_requires_new_score_room_and_position(self):
        score = dict(score_start=0., score_latest=30., room2_count=1)
        self.assertIsNotNone(handoff_reason([2.68, 2.45], score))
        self.assertIsNone(handoff_reason([2.5, 1.85], score))
        self.assertIsNone(handoff_reason([2.68, 2.45], dict(score, room2_count=0)))
        self.assertIsNone(handoff_reason([2.68, 2.45], dict(score, score_start=30.)))
        self.assertIsNone(handoff_reason([2.68, 2.45], dict(score, score_start=None)))

    def test_explicit_exit_plane_works_without_score(self):
        self.assertIsNone(handoff_reason([2.5, 2.99], {}, exit_y=3.))
        self.assertEqual(handoff_reason([2.5, 3.], {}, exit_y=3.),
                         'passed_obstacle_exit_y')
        self.assertIsNone(handoff_reason([np.nan, 3.], {}, exit_y=3.))

    def test_dqn_first_can_choose_off_route_high_q(self):
        p = policy(PreciseDQN); p.tree = cKDTree([[3., 4.]])
        start = np.array([1.5, 2.])
        path = np.array([start, [1.5, 2.05], [1.5, 2.1], [1.5, 2.15]])
        values = np.zeros(8); values[5] = 10
        target, info = p.choose_dqn_first(start, path, values, [])
        self.assertEqual(info['action'], 5)
        self.assertTrue(p.known_clear(start, target))

    def test_local_dqn_rejects_collision_and_visited_target(self):
        p = policy(PreciseDQN); start = np.array([1.5, 2.])
        p.tree = cKDTree([[1.5, 2.26]])
        values = np.zeros(8); values[0] = 100; values[1] = 90
        visited = [start + [0, -s] for s in (.025, .05, .1, .15)]
        target, info = p.choose_dqn_first(start, None, values, visited)
        self.assertNotIn(info['action'], [0, 1])
        self.assertTrue(p.known_clear(start, target))

    def test_floor_clears_ghost_but_preserves_unseen_and_current_obstacles(self):
        memory = {(500, 1000), (1000, 1000), (1500, 1000)}
        current = np.array([[3., 2.]])
        floor = np.array([[1., 2.], [3., 2.]])
        kept, removed = clear_observed_floor(memory, current, floor)
        self.assertEqual(removed, 1)
        self.assertEqual(kept, {(1000, 1000), (1500, 1000)})

    def test_route_guard_rejects_high_q_retreat(self):
        p = policy()
        values = np.zeros(8); values[1] = 100; values[0] = 2
        report = RouteGuard(p).assess(np.array([1.5, 2.]), values)
        self.assertEqual(report['raw_action'], 1)
        self.assertEqual(report['selected_action'], 0)

    def test_disconnected_goal_is_not_local_clearance(self):
        p = policy(); p.blocked[25, :] = True
        start = np.array([1.5, 2.])
        self.assertTrue(p.known_clear(start, start+[0, .15]))
        self.assertEqual(RouteGuard(p).assess(start, np.zeros(8))['status'], 'no_route')

    def test_blocked_start_reports_surface(self):
        p = policy(); p.blocked[:] = True
        report = RouteGuard(p).assess(np.array([1.5, 2.]), np.zeros(8))
        self.assertEqual(report['status'], 'start_blocked')
        self.assertTrue(all('pointcloud' in a['rejected_by'] for a in report['actions']))

    def test_precise_segment_checks_middle_not_only_endpoints(self):
        p = policy(PreciseDQN); p.tree = cKDTree([[1.5, 2.5]])
        self.assertTrue(p.known_clear([1., 2.5], [1., 2.5]))
        self.assertTrue(p.known_clear([2., 2.5], [2., 2.5]))
        self.assertFalse(p.known_clear([1., 2.5], [2., 2.5]))

    def test_precise_route_goes_around_obstacle(self):
        p = policy(PreciseDQN); p.route_resolution = .05
        p.tree = cKDTree([[1.5, 2.7]])
        start = np.array([1.5, 2.])
        path = p.plan(start)
        self.assertLessEqual(np.linalg.norm(path[-1]-p.data['goal']), .10)
        for a, b in zip(path[:-1], path[1:]):
            self.assertTrue(p.known_clear(a, b))
        target, info = p.choose(start, path, np.arange(8))
        self.assertTrue(p.known_clear(start, target))
        self.assertGreater(info['progress'], 0)

    def test_velocity_feedback_corrects_heading_at_rest(self):
        body, angular, error = velocity_command(np.zeros(2), np.zeros(2), 0., .1, 0.)
        np.testing.assert_array_equal(body, [0., 0.])
        self.assertLess(angular, 0.)

    def test_small_target_overcomes_velocity_deadband_but_stops_at_tolerance(self):
        body, _, _ = velocity_command(np.zeros(2), np.array([.01, 0]), 0, 0, 0,
                                       tolerance=.005, min_speed=.02)
        self.assertAlmostEqual(np.linalg.norm(body), .02)
        body, _, _ = velocity_command(np.zeros(2), np.array([.004, 0]), 0, 0, 0,
                                       tolerance=.005, min_speed=.02)
        np.testing.assert_array_equal(body, [0., 0.])

    def test_step_limit_is_failure(self):
        p = policy()
        rospy = types.SimpleNamespace(is_shutdown=lambda: False, loginfo=lambda *a: None,
            logwarn=lambda *a: None, Time=types.SimpleNamespace(now=lambda: 1))
        args = types.SimpleNamespace(steps=1, execute=True)
        session = (p, lambda: None, lambda: (np.array([1.5, 2.]), 0.), (None,))
        with patch.dict(sys.modules, rospy=rospy), \
             patch('task2a_cloud_step.create_session', return_value=session), \
             patch('task2a_cloud_step.run_one_step', return_value=True):
            self.assertFalse(run_steps(args))


if __name__ == '__main__':
    unittest.main()
