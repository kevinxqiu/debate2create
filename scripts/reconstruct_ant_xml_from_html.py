#!/usr/bin/env python3
"""Reconstruct the exact Ant MJCF used by a Brax HTML policy render.

A Brax HTML render embeds both the rollout trajectory and the geometry it was
generated with. When the original design XML is unavailable (or does not match
the rollout, as detected by FK validation), this script rebuilds a faithful
render XML by patching the repo Ant template with the kinematic offsets implied
by the trajectory and the capsule/sphere geometry embedded in the HTML.

Only rendering-relevant quantities are reconstructed (body offsets, geom
shapes, timestep). Actuators, masses, and joint limits are inherited from the
template and must not be treated as the trained design's true values.
"""

from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from extract_brax_html_trajectory import decode_brax_html


def _rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def implied_offsets(states: list[dict]) -> dict[int, np.ndarray]:
    """Per-link body offsets in the parent frame, checked constant over time."""
    link_pos = np.array([s["pos"] for s in states], dtype=np.float64)
    link_rot = np.array([s["rot"] for s in states], dtype=np.float64)
    parents = {1: 0, 2: 1, 3: 0, 4: 3, 5: 0, 6: 5, 7: 0, 8: 7}
    offsets: dict[int, np.ndarray] = {}
    frames = np.linspace(0, len(states) - 1, 8, dtype=int)
    for child, parent in parents.items():
        samples = []
        for frame in frames:
            parent_rot = _rotmat(link_rot[frame, parent])
            samples.append(
                parent_rot.T @ (link_pos[frame, child] - link_pos[frame, parent])
            )
        samples = np.asarray(samples)
        spread = float(np.abs(samples - samples[0]).max())
        if spread > 1e-3:
            raise SystemExit(
                f"Implied offset for link {child} is not constant (spread {spread:.2e})"
            )
        offsets[child] = samples.mean(axis=0)
    return offsets


def capsule_fromto(geom: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert an embedded capsule (pos/rot/size) to local fromto + radius."""
    pos = np.asarray(geom["pos"], dtype=np.float64)
    axis = _rotmat(np.asarray(geom["rot"], dtype=np.float64)) @ np.array(
        [0.0, 0.0, 1.0]
    )
    radius = float(geom["size"][0])
    half_length = float(geom["size"][1])
    return pos - half_length * axis, pos + half_length * axis, radius


def _fmt(vec: np.ndarray) -> str:
    return " ".join(f"{v:.6g}" for v in np.asarray(vec, dtype=np.float64))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", required=True, help="Brax HTML render file.")
    parser.add_argument("--template", required=True, help="Ant MJCF template to patch.")
    parser.add_argument("--output", required=True, help="Output XML path.")
    parser.add_argument(
        "--body-rgb",
        nargs=3,
        type=float,
        help="Optional flat body color override (r g b in 0-1).",
    )
    args = parser.parse_args()

    html_path = Path(args.html).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"Output exists, refusing to overwrite: {output}")

    system = decode_brax_html(html_path)
    states = system["states"]["x"]
    offsets = implied_offsets(states)
    geoms = system["geoms"]

    tree = ET.parse(Path(args.template).expanduser())
    root = tree.getroot()
    root.find("option").set("timestep", f"{float(system['opt']['timestep']):.6g}")

    torso = root.find("worldbody/body[@name='torso']")
    torso_geoms = geoms["torso"]
    sphere = next(g for g in torso_geoms if g["name"] == "Sphere")
    torso_capsules = [g for g in torso_geoms if g["name"] == "Capsule"]

    sphere_elem = next(g for g in torso.findall("geom") if g.get("type") == "sphere")
    sphere_elem.set("size", f"{float(sphere['size'][0]):.6g}")

    for leg_idx, leg_body in enumerate(torso.findall("body"), start=1):
        aux_link = 2 * leg_idx - 1
        ankle_link = 2 * leg_idx
        aux_off = offsets[aux_link]
        ankle_off = offsets[ankle_link]

        # Torso-level capsule for this leg, matched by xy sign pattern.
        signs = np.sign(aux_off[:2])
        upper = next(
            g
            for g in torso_capsules
            if np.all(np.sign(np.asarray(g["pos"][:2])) == signs)
        )
        c_from, c_to, radius = capsule_fromto(upper)
        leg_geom = leg_body.find("geom")
        leg_geom.set("fromto", f"{_fmt(c_from)} {_fmt(c_to)}")
        leg_geom.set("size", f"{radius:.6g}")

        aux_body = leg_body.find("body")
        aux_body.set("pos", _fmt(aux_off))
        aux_geom_json = geoms[f"aux_{leg_idx}"]
        aux_geom_json = (
            aux_geom_json[0] if isinstance(aux_geom_json, list) else aux_geom_json
        )
        c_from, c_to, radius = capsule_fromto(aux_geom_json)
        aux_geom = aux_body.find("geom")
        aux_geom.set("fromto", f"{_fmt(c_from)} {_fmt(c_to)}")
        aux_geom.set("size", f"{radius:.6g}")

        ankle_body = aux_body.find("body")
        ankle_body.set("pos", _fmt(ankle_off))
        ankle_json = geoms[f"link {ankle_link}"]
        ankle_json = ankle_json[0] if isinstance(ankle_json, list) else ankle_json
        c_from, c_to, radius = capsule_fromto(ankle_json)
        ankle_geom = ankle_body.find("geom")
        ankle_geom.set("fromto", f"{_fmt(c_from)} {_fmt(c_to)}")
        ankle_geom.set("size", f"{radius:.6g}")

    if args.body_rgb:
        r, g, b = args.body_rgb
        for texture in root.findall("asset/texture"):
            if texture.get("name") == "texgeom":
                texture.set("rgb1", f"{r:.3g} {g:.3g} {b:.3g}")
                texture.set("rgb2", f"{r:.3g} {g:.3g} {b:.3g}")
        for default_geom in root.findall("default/geom"):
            default_geom.set("rgba", f"{r:.3g} {g:.3g} {b:.3g} 1")

    output.parent.mkdir(parents=True, exist_ok=True)
    header = ET.Comment(
        " Reconstructed render XML: kinematic offsets and geom shapes recovered "
        f"from {html_path.name}; non-visual parameters inherited from template. "
    )
    root.insert(0, copy.copy(header))
    tree.write(output, encoding="unicode", xml_declaration=False)
    print(f"Wrote reconstructed XML: {output}")


if __name__ == "__main__":
    main()
