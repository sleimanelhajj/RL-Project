"""Matched validation ablation of conditional transit-scan utility."""
import csv
import numpy as np
from flight import Flight
from lookahead_search import PathSearch
from mission import CENTERS
from optimize_sweep import OUT, run


class RouteScore(PathSearch):
    def __init__(self, n, weights, mode):
        super().__init__(n, weights)
        self.mode = mode
        self.midpoint_coefficient = self.midpoint_weight

    def waypoint_score(self, w, log_mass, distance, travel):
        if self.mode == "eligible":
            midpoint = (w.flight.pos[..., None, :2] + CENTERS) * .5
            from_last = np.linalg.norm(midpoint - self.last[..., None, :], axis=-1)
            from_forecast = np.linalg.norm(midpoint - w.forecast()[:, None, None, :], axis=-1)
            eligible = (from_last > self.spacing[..., None]) & (from_forecast < 13)
            self.midpoint_weight = self.midpoint_coefficient * eligible
        elif self.mode == "zero":
            self.midpoint_weight = 0.
        return super().waypoint_score(w, log_mass, distance, travel)


def trial(mode):
    rng = np.random.default_rng(521)
    frozen = np.load(OUT / "transit-weights.npy")
    mean = np.array([.6, .45, .35])
    spread = np.array([.4, .25, .2])
    for generation in range(2):
        samples = np.clip(rng.normal(mean, spread, (6, 3)), [.05, .1, .02], [2, 1.5, 1.4])
        samples[0] = mean
        params = np.column_stack([np.tile(frozen, (len(samples), 1)), samples])
        seeds = np.arange(3160000, 3160030) + generation * 100
        ids = np.tile(seeds, len(samples))
        a = run(ids, RouteScore(len(ids), np.repeat(params, len(seeds), axis=0), mode), Flight)
        score = (a[:, 3] - .004 * a[:, 2]).reshape(len(samples), len(seeds)).mean(1)
        mean = samples[np.argsort(score)[-2:]].mean(0)
        spread *= .7
    params = np.r_[frozen, mean]
    ids = np.arange(3170000, 3170200)
    a = run(ids, RouteScore(len(ids), params, mode), Flight)
    np.save(OUT / f"route-ablation-{mode}.npy", params)
    np.savetxt(OUT / f"route-ablation-{mode}.csv", np.column_stack([ids, a]), delimiter=",",
               header="seed,found,crash,capped_steps,return", comments="")
    return [mode, int(a[:, 0].sum()), int(a[:, 1].sum()), a[:, 2].mean()*.5, (a[:, 3]-.004*a[:, 2]).mean()]


if __name__ == "__main__":
    with (OUT / "route-ablation.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy", "detected", "crashes", "capped_seconds", "objective"])
        for mode in ("legacy", "zero", "eligible"):
            row = trial(mode)
            writer.writerow(row)
            handle.flush()
            print(row, flush=True)
