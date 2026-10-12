"""Проверки правил боя и корректности обновления Q-таблицы."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class LearningContract(unittest.TestCase):
    def test_module_exists(self):
        self.assertTrue((Path(__file__).resolve().parents[1] / "learning.py").exists(),
                        "Не реализована среда боя и Q-обучение")

    def load(self):
        import learning
        return learning

    def test_victory_prevents_enemy_counterattack(self):
        m = self.load()
        env = m.BattleEnv(42)
        env.state = (1, 2, 0, 0, 1)
        state, reward, done, info = env.step(m.ATTACK)
        self.assertEqual(state[0:2], (1, 0))
        self.assertTrue(done)
        self.assertEqual(info["outcome"], "victory")
        self.assertEqual(reward, 43)

    def test_healing_is_capped_and_potion_is_consumed(self):
        m = self.load()
        env = m.BattleEnv(42)
        env.state = (5, 8, 1, 1, 0)
        state, reward, done, info = env.step(m.HEAL)
        self.assertEqual(state[:4], (5, 8, 1, 0))
        self.assertEqual(info["healed"], 1)
        self.assertFalse(done)
        self.assertEqual(reward, -4)

    def test_guard_blocks_strong_attack_and_restores_energy(self):
        m = self.load()
        env = m.BattleEnv(42)
        env.state = (2, 8, 2, 0, 1)
        state, reward, done, info = env.step(m.GUARD)
        self.assertEqual(state[:4], (2, 8, 3, 0))
        self.assertEqual(info["received"], 0)
        self.assertEqual(reward, -1)

    def test_invalid_heavy_strike_still_costs_enemy_turn(self):
        m = self.load()
        env = m.BattleEnv(42)
        env.state = (3, 8, 1, 0, 1)
        state, reward, done, info = env.step(m.HEAVY)
        self.assertEqual(state[:4], (0, 8, 1, 0))
        self.assertTrue(done)
        self.assertTrue(info["invalid"])
        self.assertEqual(reward, -55)

    def test_terminal_target_has_no_future_value(self):
        m = self.load()
        q = {(1, 1, 0, 0, 0): [0.0]*4, (0, 1, 0, 0, 0): [999.0]*4}
        m.q_update(q, (1, 1, 0, 0, 0), 0, -40, (0, 1, 0, 0, 0),
                   terminated=True, alpha=0.2, gamma=0.95)
        self.assertEqual(q[(1, 1, 0, 0, 0)][0], -8)

    def test_time_limit_does_not_zero_bootstrap(self):
        m = self.load()
        q = {(1, 1, 0, 0, 0): [0.0]*4, (1, 1, 1, 0, 0): [10.0]*4}
        m.q_update(q, (1, 1, 0, 0, 0), 0, -1, (1, 1, 1, 0, 0),
                   terminated=False, alpha=1, gamma=0.9)
        self.assertEqual(q[(1, 1, 0, 0, 0)][0], 8)

    def test_step_after_terminal_is_rejected(self):
        m = self.load()
        env = m.BattleEnv(42)
        env.state = (0, 5, 0, 0, 0)
        with self.assertRaises(RuntimeError):
            env.step(m.ATTACK)

    def test_seed_reproduces_training(self):
        m = self.load()
        q1, log1, _ = m.train(episodes=100, seed=42, checkpoint_every=0)
        q2, log2, _ = m.train(episodes=100, seed=42, checkpoint_every=0)
        self.assertEqual(q1, q2)
        self.assertEqual(log1, log2)

if __name__ == "__main__":
    unittest.main()
