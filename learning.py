"""Q-обучение для выбора действий персонажа в пошаговом бою.

Только стандартная библиотека Python. Обучение и оценка работают без GUI.
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

ATTACK, HEAVY, GUARD, HEAL = range(4)
ACTION_NAMES = ("Атака", "Сильный удар", "Защита", "Лечение")
MAX_HP, MAX_ENERGY, MAX_STEPS = 6, 3, 30
ALPHA, GAMMA = 0.2, 0.95


class BattleEnv:
    """Состояние s = (здоровье, враг, энергия, зелье, намерение)."""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.state = None

    def reset(self):
        # Разные старты учат реагировать на ситуацию, а не запоминать бой.
        self.state = (
            self.rng.randint(2, 6), self.rng.randint(6, 8),
            self.rng.randint(0, 3), self.rng.randint(0, 1),
            int(self.rng.random() < 0.4),
        )
        return self.state

    def step(self, action):
        if action not in range(4):
            raise ValueError("Неизвестное действие")
        if self.state is None or min(self.state[:2]) <= 0:
            raise RuntimeError("Сначала начните новый эпизод")
        hp, enemy, energy, potion, intent = self.state
        dealt = received = healed = 0
        invalid = False

        # Сначала выполняется действие персонажа.
        if action == ATTACK:
            dealt = min(2, enemy)
            energy = min(MAX_ENERGY, energy + 1)
        elif action == HEAVY:
            if energy >= 2:
                dealt = min(4, enemy)
                energy -= 2
            else:
                invalid = True
        elif action == GUARD:
            energy = min(MAX_ENERGY, energy + 1)
        elif potion:
            healed = min(4, MAX_HP - hp)
            hp += healed
            potion = 0
        else:
            invalid = True
        enemy -= dealt

        # Побеждённый враг уже не отвечает ударом.
        if enemy > 0:
            damage = 3 if intent else 1
            if action == GUARD:
                damage = max(0, damage - 3)
            received = min(damage, hp)
            hp -= received

        # Защита действует один ход. Новое намерение объявляется заранее.
        next_intent = int(self.rng.random() < 0.4)
        self.state = (hp, enemy, energy, potion, next_intent)
        outcome = "victory" if enemy == 0 else "defeat" if hp == 0 else ""
        terminated = bool(outcome)

        # Награда использует фактический урон, исключая лишние очки
        # за избыточный удар или многократное лечение без зелья.
        reward = -1 + 2 * dealt - 3 * received - 5 * int(invalid)
        if outcome == "victory":
            reward += 40
        elif outcome == "defeat":
            reward -= 40
        info = dict(outcome=outcome, dealt=dealt, received=received,
                    healed=healed, invalid=invalid)
        return self.state, reward, terminated, info


def greedy_action(q, state, rng):
    # При равных Q выбор случаен: номер действия не даёт преимущества.
    values = q.get(state, [0.0] * 4)
    best = max(values)
    return rng.choice([a for a, v in enumerate(values) if v == best])


def q_update(q, state, action, reward, next_state,
             terminated, alpha=ALPHA, gamma=GAMMA):
    """Одна формула Беллмана; терминальный переход не имеет продолжения."""
    values = q.setdefault(state, [0.0] * 4)
    future = 0.0 if terminated else max(q.get(next_state, [0.0] * 4))
    target = reward + gamma * future
    values[action] += alpha * (target - values[action])


def epsilon_at(episode, episodes):
    # Линейное уменьшение исследования до 0.05 за первые 80% обучения.
    progress = min(1.0, episode / max(1, int(0.8 * episodes) - 1))
    return 1.0 - 0.95 * progress


def evaluate(q, episodes=1000, seed=100000, random_policy=False):
    """Новые бои. Каждому номеру эпизода соответствует свой seed среды."""
    records = []
    for episode in range(episodes):
        env = BattleEnv(seed + episode)
        state = env.reset()
        rng = random.Random(seed + 1000000 + episode)
        total, outcome, invalid_count = 0, "timeout", 0
        for step in range(1, MAX_STEPS + 1):
            action = rng.randrange(4) if random_policy else (
                greedy_action(q, state, rng))
            state, reward, done, info = env.step(action)
            total += reward
            invalid_count += int(info["invalid"])
            if done:
                outcome = info["outcome"]
                break
        records.append(dict(episode=episode + 1, outcome=outcome,
                            reward=total, steps=step,
                            invalid_actions=invalid_count))
    wins = sum(x["outcome"] == "victory" for x in records)
    # Интервал Уилсона: оценка точности доли побед на выборке.
    n, z = episodes, 1.96
    p = wins / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    summary = dict(
        episodes=n, wins=wins, win_rate=p,
        defeats=sum(x["outcome"] == "defeat" for x in records),
        timeouts=sum(x["outcome"] == "timeout" for x in records),
        mean_reward=sum(x["reward"] for x in records)/n,
        mean_steps=sum(x["steps"] for x in records)/n,
        invalid_actions=sum(x["invalid_actions"] for x in records),
        win_ci95=[center-radius, center+radius],
    )
    return summary, records


def train(episodes=20000, seed=42, checkpoint_every=2000):
    q, history, checkpoints = {}, [], []
    env, rng = BattleEnv(seed), random.Random(seed + 1)
    if checkpoint_every:
        summary, _ = evaluate(q, episodes=500)
        checkpoints.append(dict(episode=0, **summary))
    for episode in range(episodes):
        state = env.reset()
        epsilon = epsilon_at(episode, episodes)
        total, outcome = 0, "timeout"
        for step in range(1, MAX_STEPS + 1):
            # epsilon-greedy совмещает поиск и использование опыта.
            action = rng.randrange(4) if rng.random() < epsilon else (
                greedy_action(q, state, rng))
            next_state, reward, done, info = env.step(action)
            q_update(q, state, action, reward, next_state, done)
            state, total = next_state, total + reward
            if done:
                outcome = info["outcome"]
                break
        # 30 ходов - внешнее усечение, а не терминальное состояние MDP.
        # Поэтому на таком переходе выше сохраняется bootstrap.
        history.append(dict(episode=episode+1, reward=total, steps=step,
                            epsilon=epsilon, outcome=outcome))
        if checkpoint_every and (episode+1) % checkpoint_every == 0:
            summary, _ = evaluate(q, episodes=500)
            checkpoints.append(dict(episode=episode+1, **summary))
    return q, history, checkpoints


def demonstration(q, seed=15, initial=(1, 8, 0, 1, 0)):
    """Показ боя из сложного достижимого состояния с низким здоровьем."""
    env, rng = BattleEnv(seed), random.Random(seed + 1000)
    env.state = initial
    frames, total = [], 0
    for turn in range(1, MAX_STEPS + 1):
        state = env.state
        action = greedy_action(q, state, rng)
        next_state, reward, done, info = env.step(action)
        total += reward
        frames.append(dict(turn=turn, state=list(state), action=action,
                           action_name=ACTION_NAMES[action],
                           next_state=list(next_state), reward=reward,
                           q_values=q.get(state, [0.0]*4),
                           **info))
        if done:
            break
    return dict(seed=seed, initial=list(initial), turns=frames,
                total_reward=total,
                outcome=frames[-1]["outcome"] or "timeout")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes должен быть положительным")
    args.out.mkdir(parents=True, exist_ok=True)
    q, history, checkpoints = train(args.episodes, args.seed)
    baseline, baseline_rows = evaluate(q, random_policy=True)
    learned, learned_rows = evaluate(q)
    demo = demonstration(q)
    parameters = dict(episodes=args.episodes, seed=args.seed,
                      alpha=ALPHA, gamma=GAMMA, epsilon_start=1.0,
                      epsilon_end=0.05, max_steps=MAX_STEPS,
                      evaluation_seed=100000)
    results = dict(parameters=parameters, baseline=baseline,
                   learned=learned, q_states=len(q),
                   demo=demo, checkpoints=checkpoints)
    (args.out / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    # JSON сохраняет все четыре оценки действий каждого посещённого s.
    (args.out / "q_table.json").write_text(
        json.dumps({",".join(map(str, s)): v for s, v in sorted(q.items())},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out / "training.csv", history)
    write_csv(args.out / "evaluation_random.csv", baseline_rows)
    write_csv(args.out / "evaluation_learned.csv", learned_rows)
    print(json.dumps({k: results[k] for k in
                      ("parameters", "baseline", "learned", "q_states")},
                     ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
