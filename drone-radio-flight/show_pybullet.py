"""Record actual PyBullet frames of a scripted maritime search mission."""
import argparse
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

from bullet_flight import BulletFlight, PALETTE
from flight import DT, TICKS
from mission import LINKS, RADIUS, SIZE, Worlds, greedy

VIEW_W, VIEW_H = 830, 560
OFFSET = (24, 100)
CANVAS = (1160, 710)
WHITE = "#ecf5ff"
MUTED = "#9fb3c9"
LINK_UP = "#66dcb6"
LINK_DOWN = "#ff9c9d"
COLORS = [tuple(round(c * 255) for c in rgb[:3]) for rgb in PALETTE]
FONTS = Path("C:/Windows/Fonts")


def typography(size, bold=False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = FONTS / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default(size=size)


TITLE, BODY, SMALL, LARGE = (typography(29, True), typography(17), typography(13), typography(24, True))


class ReplayCamera:
    def __init__(self, client, initial_target):
        self.client = client
        self.view = p.computeViewMatrixFromYawPitchRoll([21, 21, 0], 57, 0, -58, 0, 2)
        self.projection = p.computeProjectionMatrixFOV(47, VIEW_W / VIEW_H, .2, 130)
        self.matrix = (np.asarray(self.projection).reshape(4, 4, order="F") @
                       np.asarray(self.view).reshape(4, 4, order="F"))
        for extent, z, color in ((300, -.02, [.035, .10, .17, 1]),
                                 (SIZE / 2, .01, [.07, .23, .36, 1])):
            surface = p.createVisualShape(p.GEOM_BOX, halfExtents=[extent, extent, .05],
                                          rgbaColor=color, physicsClientId=client)
            p.createMultiBody(0, baseVisualShapeIndex=surface,
                              basePosition=[SIZE / 2, SIZE / 2, z], physicsClientId=client)
        marker = p.createVisualShape(p.GEOM_SPHERE, radius=.28, rgbaColor=[1, .32, .37, 1],
                                     physicsClientId=client)
        self.target = p.createMultiBody(0, baseVisualShapeIndex=marker,
                                        basePosition=[*initial_target, .20], physicsClientId=client)

    def project(self, pos):
        x, y, _, w = self.matrix @ [*pos, 1]
        return ((x / w + 1) * VIEW_W / 2 + OFFSET[0],
                (1 - y / w) * VIEW_H / 2 + OFFSET[1])

    def frame(self, world, seed, trails, search):
        target = world.target()[0]
        p.resetBasePositionAndOrientation(self.target, [*target, .20], [0, 0, 0, 1],
                                          physicsClientId=self.client)
        _, _, rgba, _, _ = p.getCameraImage(VIEW_W, VIEW_H, self.view, self.projection,
                                            lightDirection=[-1, -1, 4], shadow=0,
                                            renderer=p.ER_TINY_RENDERER, physicsClientId=self.client)
        scene = np.asarray(rgba, dtype=np.uint8).reshape(VIEW_H, VIEW_W, 4)
        frame = Image.new("RGB", CANVAS, "#101d30")
        frame.paste(Image.fromarray(scene).convert("RGB"), OFFSET)
        draw = ImageDraw.Draw(frame)
        draw.text((24, 20), "Three-drone search mission", fill=WHITE, font=TITLE)
        draw.text((25, 62), f"PYBULLET 3.2.7  /  RADIO-AWARE GREEDY  /  SCENARIO {seed}",
                  fill=MUTED, font=SMALL)
        for cell in range(0, SIZE + 1, 6):
            draw.line([self.project((cell, 0, .01)), self.project((cell, SIZE, .01))],
                      fill="#366079", width=1)
            draw.line([self.project((0, cell, .01)), self.project((SIZE, cell, .01))],
                      fill="#366079", width=1)
        for i, (pos, trail) in enumerate(zip(world.flight.pos[0], trails)):
            color = COLORS[i]
            if len(trail) > 1:
                draw.line([self.project((*pt[:2], .03)) for pt in trail], fill=color, width=3)
            if search[i]:
                ring = [self.project((pos[0] + RADIUS * np.cos(a), pos[1] + RADIUS * np.sin(a), .025))
                        for a in np.linspace(0, 2 * np.pi, 32)]
                draw.line(ring, fill=color, width=2)
            x, y = self.project(pos)
            lx, ly = ((x - 39, y - 21), (x + 10, y - 32), (x + 12, y + 7))[i]
            draw.line((x, y, lx + 14, ly + 9), fill=color, width=1)
            draw.rounded_rectangle((lx, ly, lx + 29, ly + 18), radius=4, fill="#14253a")
            draw.text((lx + 4, ly), f"D{i + 1}", fill=color, font=SMALL)
        x, y = self.project([*target, .26])
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline="#ff9298", width=2)
        draw.text((x + 10, y - 11), "Person", fill="#ffb9bb", font=SMALL,
                  stroke_width=2, stroke_fill="#14253a")

        draw.rounded_rectangle((874, 100, 1136, 660), radius=16, fill="#172841")
        draw.text((894, 122), f"{world.t * DT * TICKS:04.1f} s", fill=WHITE, font=LARGE)
        draw.text((894, 158), f"Mission step {world.t:02d} / 100", fill=MUTED, font=SMALL)
        status = ("PERSON FOUND" if world.found[0] else "CONTACT / CRASH" if world.crashed[0]
                  else "SEARCH IN PROGRESS" if world.active[0] else "TIME LIMIT")
        draw.text((894, 185), status, fill=LINK_UP if world.found[0] else WHITE, font=BODY)
        draw.line((894, 224, 1116, 224), fill="#324965", width=1)
        draw.text((894, 242), "Radio links", fill=WHITE, font=BODY)
        trace = world.trace[max(world.t - 1, 0), 0]
        for i, ((src, dst), blocked) in enumerate(zip(LINKS, trace)):
            y = 275 + 27 * i
            draw.ellipse((894, y + 6, 902, y + 14), fill=LINK_DOWN if blocked else LINK_UP)
            draw.text((913, y), f"D{src + 1} to D{dst + 1}", fill=MUTED, font=SMALL)
            draw.text((1042, y), "OUTAGE" if blocked else "OPEN", fill=LINK_DOWN if blocked else LINK_UP,
                      font=SMALL)
        draw.line((894, 448, 1116, 448), fill="#324965", width=1)
        draw.text((894, 469), f"Search attempts   {world.searches[0]}", fill=WHITE, font=BODY)
        draw.text((894, 501), f"Radio received   {world.received[0]:,} B", fill=WHITE, font=BODY)
        for i, color in enumerate(COLORS):
            draw.ellipse((895, 546 + i * 26, 905, 556 + i * 26), fill=color)
            draw.text((914, 541 + i * 26), f"Drone {i + 1}", fill=MUTED, font=BODY)
        draw.text((25, 679), "Actual PyBullet frames. The person marker is visible to the viewer only.",
                  fill=MUTED, font=SMALL)
        return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=90105)
    parser.add_argument("--out", type=Path, default=Path("results/pybullet-search-replay.gif"))
    args = parser.parse_args()
    world = Worlds([args.seed], ["burst25"], flight_factory=BulletFlight)
    try:
        camera = ReplayCamera(world.flight.client, world.target()[0])
        trails = [[pos.copy()] for pos in world.flight.pos[0]]
        frames = [camera.frame(world, args.seed, trails, [False] * len(trails))]
        while world.active[0]:
            actions = greedy(world, radio=True)
            world.step(actions)
            for trail, position in zip(trails, world.flight.pos[0]):
                trail.append(position.copy())
            frames.append(camera.frame(world, args.seed, trails, actions[0] == 8))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(args.out, save_all=True, append_images=frames[1:],
                       duration=[400] * (len(frames) - 1) + [2000], loop=0, disposal=2)
        print(f"seed={args.seed} steps={world.t} time={world.t * DT * TICKS:.1f}s "
              f"detected={bool(world.found[0])} crash={bool(world.crashed[0])} "
              f"searches={world.searches[0]} radio_received={world.received[0]} frames={len(frames)}")
    finally:
        world.flight.close()


if __name__ == "__main__":
    main()
