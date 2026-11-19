"""Build the results figure directly from the recorded simulation."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiment import ARMS, RADIOS, RULES, TEST_IDS, play
from mission import HORIZON, SIZE

RESULTS = Path("results")
METHODS = (*RULES, *ARMS)
LABELS = ("Sweep", "Greedy", "Radio greedy", "FF", "FF + age", "GRU", "GRU + age")
COLORS = ("#566779", "#879aac", "#b36323", "#266f91", "#6aa7c2", "#4942a4", "#a3539d")
RADIO_LABELS = ("Reliable", "Scattered", "5-step burst", "25-step burst", "Off")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#9eabb7",
                     "text.color": "#263443", "axes.labelcolor": "#405368",
                     "xtick.color": "#405368", "ytick.color": "#405368"})


def load(path):
    with path.open(encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def array(rows, method, radio, field):
    selected = sorted((int(row["seed"]), int(row["world"]), float(row[field])) for row in rows
                      if row["method"] == method and row["radio"] == radio)
    return np.array([v for _, _, v in selected]).reshape(-1, len(TEST_IDS))


def title(axis, string):
    axis.set_title(string, loc="left", fontweight="bold", pad=10)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#d9e2eb", alpha=.75, linewidth=.7)


def main():
    rows = load(RESULTS / "baselines.csv")
    for arm in range(4):
        rows += load(RESULTS / f"evaluated-{arm}.csv")
    if len(rows) != (3 + 4 * 3) * len(RADIOS) * len(TEST_IDS):
        raise ValueError("Incomplete results")
    keys = {(row["method"], row["seed"], row["radio"], row["world"]) for row in rows}
    if len(keys) != len(rows) or {int(row["world"]) for row in rows} != set(TEST_IDS):
        raise ValueError("Held-out runs must contain unique policy/seed/radio/world records")
    fig = plt.figure(figsize=(15.2, 11.7), layout="constrained", facecolor="#f3f7fa")
    grid = fig.add_gridspec(3, 2)
    a, b, c, d, e = [fig.add_subplot(grid[ix]) for ix in ((0,0),(0,1),(1,0),(1,1),(2,0))]
    flight = fig.add_subplot(grid[2, 1], projection="3d")
    fig.suptitle("Cooperative drone search under radio outages\n"
                 "3D rigid-body pilot  |  96 matched test missions  |  3 seeds per MAPPO variant",
                 fontsize=21, fontweight="bold", x=.02, ha="left")
    for axis in (a,b,c,d,e):
        axis.set_facecolor("white")

    title(a, "A   Detection by the deadline")
    for i, method in enumerate(METHODS):
        data = [100 * array(rows, method, mode, "found").mean() for mode in RADIOS]
        a.plot(RADIO_LABELS, data, marker="o", lw=2, color=COLORS[i],
               ls="--" if i < 3 else "-", label=LABELS[i])
    a.set_ylabel("Missions detected (%)")
    a.set_ylim(0, 100)
    a.legend(ncol=3, loc="lower left", frameon=False, fontsize=9)

    title(b, "B   Detection time under 25-step bursts")
    times = [.5 * array(rows, method, "burst25", "time").mean() for method in METHODS]
    yy = np.arange(len(METHODS))
    b.barh(yy, times, color=COLORS, height=.65)
    b.set_yticks(yy, LABELS)
    b.invert_yaxis()
    b.set_xlabel("Mean seconds  •  Misses count as 50 s")
    b.set_xlim(0, 54)
    for i, n in enumerate(times):
        b.text(n+.55, i, f"{n:.1f}", va="center", fontsize=9)
    b.grid(axis="x", color="#d9e2eb", linewidth=.7)
    b.grid(axis="y", visible=False)

    title(c, "C   Change from scattered loss to 25-step bursts")
    c.axvline(0, color="#657688", lw=1)
    rng = np.random.default_rng(19)
    selection = rng.integers(0, len(TEST_IDS), (3000, len(TEST_IDS)))
    for i, method in enumerate(METHODS):
        paired = 100 * (array(rows, method, "burst25", "found") -
                        array(rows, method, "scattered", "found")).mean(axis=0)
        low, high = np.quantile(paired[selection].mean(axis=1), [.025, .975])
        mean = paired.mean()
        c.plot([low, high], [i, i], color=COLORS[i], lw=2)
        c.scatter(mean, i, color=COLORS[i], zorder=4)
    c.set_yticks(yy, LABELS)
    c.invert_yaxis()
    c.set_xlabel("Detection difference (percentage points)  •  Paired scenario intervals")
    c.grid(axis="y", visible=False)

    title(d, "D   Learning check on separate validation missions")
    positions = np.arange(len(ARMS))
    for i, method in enumerate(ARMS):
        v = json.loads((RESULTS / f"validation-start-{i}.json").read_text(encoding="utf8"))
        initial = [100 * x["initial"] / x["worlds"] for x in v]
        final = [100 * x["final"] / x["worlds"] for x in v]
        for before, after in zip(initial, final):
            d.plot([i-.12,i+.12], [before,after], color=COLORS[i+3], alpha=.32)
        d.scatter(i-.12, np.mean(initial), color="#9cabb9", s=56,
                  label="Random initialization" if i == 0 else None, zorder=3)
        d.scatter(i+.12, np.mean(final), color=COLORS[i+3], s=64,
                  label="After training" if i == 0 else None, zorder=4)
    d.set_xticks(positions, LABELS[3:])
    d.set_ylim(0, 100)
    d.set_ylabel("Missions detected (%)")
    d.legend(loc="lower right", frameon=False, fontsize=9)

    title(e, "E   MAPPO training batches")
    for i, method in enumerate(ARMS):
        train = load(RESULTS / f"training-{i}.csv")
        rounds = sorted({int(row["update"]) for row in train})
        points = np.array([[float(row["found"]) for row in train if int(row["update"]) == n]
                           for n in rounds]) * 100
        mean = points.mean(axis=1)
        smoothed = np.convolve(np.pad(mean, (1,1), mode="edge"), np.ones(3) / 3, "valid")
        e.plot(rounds, smoothed, color=COLORS[i+3], lw=2, label=LABELS[i+3])
    e.set_ylabel("Batch detection (%)")
    e.set_xlabel("Policy update  •  3-update moving average")
    e.set_ylim(0, 100)
    e.legend(ncol=2, frameon=False, fontsize=9, loc="lower right")

    flight.set_title("F   A successful held-out 3D flight", loc="left", fontweight="bold", pad=0)
    picked = next(int(r["world"]) for r in rows if r["method"] == RULES[2]
                  and r["radio"] == "burst25" and r["found"] == "1" and int(r["time"]) >= 24)
    meta, replay = play(RULES[2], None, [picked], "burst25", -1, replay=True)
    trajectory, target = replay
    for i, color in enumerate(("#007b9b", "#a74b29", "#6044af")):
        line = trajectory[:, 0, i]
        flight.plot(*line.T, color=color, lw=2, label=f"Drone {i+1}")
        flight.scatter(*line[0], color=color, marker="o", s=14)
    contact = target[-1, 0]
    flight.scatter(contact[0], contact[1], .15, color="#0a121d", marker="*", s=90, label="Person")
    flight.set(xlabel="East (m)", ylabel="North (m)", zlabel="Alt. (m)",
               xlim=(0,SIZE), ylim=(0,SIZE), zlim=(0,6))
    flight.set_zticks([0, 2, 4, 6])
    flight.set_box_aspect((1.15, 1.15, .36))
    flight.view_init(26, -55)
    flight.legend(loc="upper left", bbox_to_anchor=(-.02,.93), frameon=False, fontsize=8)
    flight.text2D(.03,.02, f"Radio-aware greedy  •  Detection at {meta[0]['time']*.5:.1f} s",
                  transform=flight.transAxes, fontsize=9)

    result_file = RESULTS / "drone-search-comparisons.png"
    fig.savefig(result_file, dpi=155)
    print(result_file)
    for method in METHODS:
        print(method, [round(100 * array(rows, method, mode, "found").mean(), 1) for mode in RADIOS])
    for i, method in enumerate(ARMS):
        v = json.loads((RESULTS / f"validation-start-{i}.json").read_text(encoding="utf8"))
        n = sum(x["worlds"] for x in v)
        print("VALID", method, round(100 * sum(x["initial"] for x in v) / n, 1),
              round(100 * sum(x["final"] for x in v) / n, 1))
    print("CRASHES", sum(int(row["crashed"]) for row in rows), "rows", len(rows))


if __name__ == "__main__":
    main()
