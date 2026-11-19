"""Reproducible shared-actor MAPPO and paired evaluation."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from mission import DRONES, HORIZON, Sweep, Worlds, greedy

RADIOS = ("reliable", "scattered", "burst5", "burst25", "off")
ARMS = {"FF": (False, False), "FF + age": (False, True),
        "GRU": (True, False), "GRU + age": (True, True)}
RULES = ("Sector sweep", "Coordinated greedy", "Radio-aware greedy")
TRAIN_RADIOS = RADIOS[:3]
TEST_IDS = np.arange(90_000, 90_096)
WIDTH = 48


class Actor(nn.Module):
    def __init__(self, recurrent, age):
        super().__init__()
        self.age = age
        self.embed = nn.Sequential(nn.Linear(53, WIDTH), nn.Tanh())
        self.memory = nn.GRU(WIDTH, WIDTH) if recurrent else None
        self.ff = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.Tanh()) if not recurrent else None
        self.out = nn.Linear(WIDTH, 9)
        nn.init.orthogonal_(self.out.weight, gain=.035)
        nn.init.zeros_(self.out.bias)

    def forward(self, sequence, hidden=None):
        if not self.age:
            sequence = torch.cat([sequence[..., :-9], torch.zeros_like(sequence[..., -9:])], -1)
        shape = sequence.shape[:-1]
        z = self.embed(sequence).reshape(shape[0], -1, WIDTH)
        if self.memory:
            z, hidden = self.memory(z, hidden)
        else:
            z = self.ff(z)
        logits = self.out(z).reshape(*shape, 9)
        # Prevent goals outside the common operating area.
        return logits.masked_fill(sequence[..., 17:26] < .5, -1e8), hidden


class Critic(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(25, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))


@torch.no_grad()
def play(name, actor, ids, radio, seed, replay=False, stochastic=False):
    world = Worlds(ids, [radio] * len(ids))
    sweep = Sweep(len(ids)) if name == RULES[0] else None
    hidden = None
    track = [world.flight.pos.copy()]
    target = [world.target().copy()]
    while world.active.any():
        if actor is not None:
            logits, hidden = actor(torch.from_numpy(world.features())[None], hidden)
            if stochastic:
                action = Categorical(logits=logits[0]).sample().numpy()
            else:
                action = logits[0].argmax(-1).numpy()
        elif sweep is not None:
            action = sweep(world)
        else:
            action = greedy(world, radio=name == RULES[2])
        world.step(action)
        if replay:
            track.append(world.flight.pos.copy())
            target.append(world.target().copy())
    rows = [dict(method=name, seed=seed, radio=radio, world=int(s), found=int(world.found[i]),
                 time=int(world.time[i]), duplicate=int(world.duplicate[i]),
                 searches=int(world.searches[i]), coverage=int(world.team[i].sum()),
                 sent=int(world.sent[i]), received=int(world.received[i]),
                 crashed=int(world.crashed[i]), reward=float(world.returns[i]))
            for i, s in enumerate(ids)]
    return rows, (np.stack(track), np.stack(target)) if replay else None


def experiment(name, seed, updates, batch, path, initial_actor=None):
    torch.manual_seed(seed)
    random = np.random.default_rng(seed)
    actor = Actor(*ARMS[name])
    if initial_actor is not None:
        actor.load_state_dict(initial_actor)
    critic = Critic()
    parameters = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(parameters, lr=.0016, eps=1e-5)
    curve = []
    for update in range(updates):
        ids = random.integers(0, 70_000, size=batch)
        radio = random.choice(TRAIN_RADIOS, size=batch)
        world = Worlds(ids, radio)
        obs, context, actions, logs, rewards, values, ends, masks = ([] for _ in range(8))
        hidden = None
        with torch.no_grad():
            while world.active.any():
                original = world.features()
                state = torch.from_numpy(world.critic(original))
                current = torch.from_numpy(original)
                logits, hidden = actor(current[None], hidden)
                dist = Categorical(logits=logits[0])
                action = dist.sample()
                mask = torch.from_numpy(world.active.copy())
                reward, done = world.step(action.numpy())
                obs.append(current)
                context.append(state)
                actions.append(action)
                logs.append(dist.log_prob(action))
                rewards.append(torch.from_numpy(reward))
                values.append(critic(state).squeeze(-1))
                ends.append(torch.from_numpy(done))
                masks.append(mask)
        obs, context, actions, logs, rewards, values, ends, masks = map(
            torch.stack, (obs, context, actions, logs, rewards, values, ends, masks))
        advantage = torch.zeros_like(rewards)
        future = torch.zeros(batch)
        value_after = torch.zeros(batch)
        for step in range(len(rewards) - 1, -1, -1):
            continuation = (~ends[step]).float()
            delta = rewards[step] + .995 * continuation * value_after - values[step]
            future = (delta + .995 * .95 * continuation * future) * masks[step]
            advantage[step] = future
            value_after = values[step]
        returns = advantage + values
        usable = advantage[masks]
        advantage = (advantage - usable.mean()) / usable.std().clamp_min(1e-6)
        for _ in range(4):
            for group in torch.randperm(batch).split(8):
                pick = masks[:, group]
                if not pick.any():
                    continue
                logits, _ = actor(obs[:, group])
                dist = Categorical(logits=logits)
                ratio = (dist.log_prob(actions[:, group]) - logs[:, group]).exp()
                gae = advantage[:, group, None]
                clipped = torch.minimum(ratio * gae, ratio.clamp(.8, 1.2) * gae)
                pred = critic(context[:, group]).squeeze(-1)
                loss = -clipped[pick].mean() + .5 * (pred[pick] - returns[:, group][pick]).square().mean()
                loss -= .025 * dist.entropy()[pick].mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, .5)
                optimizer.step()
        curve.append(dict(method=name, seed=seed, update=update + 1, found=float(world.found.mean()),
                          reward=float(world.returns.mean()), active_world_steps=int(masks.sum())))
        if (update + 1) % 5 == 0 or update + 1 == updates:
            print(name, seed, update + 1, "training detection", round(float(world.found.mean()), 3), flush=True)
    torch.save(actor.state_dict(), path / f"policy-{list(ARMS).index(name)}-{seed}.pt")
    return actor, curve


def output_csv(path, rows):
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["baselines", "train", "evaluate"])
    parser.add_argument("--updates", type=int, default=18)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--arm", type=int, choices=range(4))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    indexes = range(4) if args.arm is None else [args.arm]
    if args.stage == "baselines":
        results = []
        for method in RULES:
            for radio in RADIOS:
                rows, _ = play(method, None, TEST_IDS, radio, -1)
                results += rows
                print(method, radio, sum(r["found"] for r in rows), "/", len(rows), flush=True)
        output_csv(output / "baselines.csv", results)
    if args.stage == "train":
        for idx in indexes:
            name = list(ARMS)[idx]
            curve = []
            for seed in (101, 202, 303):
                _, records = experiment(name, seed, args.updates, args.batch, output)
                curve += records
            output_csv(output / f"training-{idx}.csv", curve)
    if args.stage == "evaluate":
        for idx in indexes:
            name = list(ARMS)[idx]
            results = []
            initial_checks = []
            validation_ids = np.arange(72_000, 72_024)
            for seed in (101, 202, 303):
                torch.manual_seed(seed)
                actor = Actor(*ARMS[name])
                torch.manual_seed(seed + 477)
                init, _ = play(name, actor, validation_ids, "burst25", seed, stochastic=True)
                actor.load_state_dict(torch.load(output / f"policy-{idx}-{seed}.pt", map_location="cpu"))
                actor.eval()
                torch.manual_seed(seed + 477)
                learned, _ = play(name, actor, validation_ids, "burst25", seed, stochastic=True)
                initial_checks.append(dict(seed=seed, initial=sum(r["found"] for r in init),
                                           final=sum(r["found"] for r in learned), worlds=len(init)))
                for radio in RADIOS:
                    torch.manual_seed(seed + 971)
                    rows, _ = play(name, actor, TEST_IDS, radio, seed, stochastic=True)
                    results += rows
                    print(name, seed, radio, sum(r["found"] for r in rows), "/", len(rows), flush=True)
            output_csv(output / f"evaluated-{idx}.csv", results)
            (output / f"validation-start-{idx}.json").write_text(json.dumps(initial_checks), encoding="utf8")


if __name__ == "__main__":
    main()

