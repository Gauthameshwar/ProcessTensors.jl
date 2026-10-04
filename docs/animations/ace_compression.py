# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: docs/animations/ace_compression.py
# Contributor: Gauthameshwar S.
#
# Generates the schematic ACE animation: two bath modes over three time steps are
# joined and compressed right to left into a three-core process tensor.
#
# Run with:
# PT_ANIM_THEME=light docs/animations/.manim_env/bin/python docs/animations/ace_compression.py
"""ACE compression animation for the ProcessTensors.jl homepage and README.

Schematic only: no SVDs, no package calls. Time runs right to left; ``k = 0`` is
the rightmost (earliest) column. Single wires are Liouville-space legs, so the
initial-condition triangles are vectorised bath density operators: they point
right, and the bath leg meets the base on the left. The latest-time bath legs
are closed by filled markers as soon as those propagators are drawn. The clip shows the initial join/compress
sweep only, not every compression schedule or canonical gauge.

Tested with Manim Community v0.21.0 (Cairo renderer), Python 3.12, ffmpeg 8.0.1.

Render from the repository root::

    PT_ANIM_THEME=light docs/animations/.manim_env/bin/python docs/animations/ace_compression.py
    PT_ANIM_THEME=dark  docs/animations/.manim_env/bin/python docs/animations/ace_compression.py

``PT_ANIM_PRESET=draft`` writes an MP4 and a poster PNG into
``docs/animations/media/drafts/``. The final preset writes a transparent GIF
and a transparent poster PNG into ``docs/src/assets/animations/``.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    AnimationGroup,
    CapStyleType,
    Circle,
    Create,
    DrawBorderThenFill,
    FadeOut,
    GrowFromCenter,
    MovingCameraScene,
    Polygon,
    ReplacementTransform,
    RoundedRectangle,
    Succession,
    Text,
    UpdateFromAlphaFunc,
    VGroup,
    VMobject,
    config,
    smooth,
    tempconfig,
)
from manim.utils.exceptions import EndSceneEarlyException

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

SCENE_STEM = "ace"
MODES = ("E1", "E2")

STYLE = {
    "outline_px": 5.0,
    "wire_px": 4.5,
    "marker_px": 4.5,
    "emphasis_px": 7.5,
    "reference_pixel_width": 960,
}

PALETTES = {
    "light": {
        "wire": "#3a3a3a",
        "input": "#138a8a",
        "output": "#c27c0e",
        "mode_1": ("#cfe3f5", "#2f6fa8"),
        "mode_2": ("#f3d3df", "#a8406b"),
        "bath_ic": ("#d4ead4", "#2f7a3e"),
        "work": ("#dcd6f2", "#4f46a0"),
        "carry": ("#dcd6f2", "#4f46a0"),
        "pt": ("#d4ead4", "#217a3e"),
        "glow": "#f2a900",
        "debug": "#7a7a7a",
    },
    "dark": {
        "wire": "#cfcfcf",
        "input": "#4fd1c5",
        "output": "#f2b544",
        "mode_1": ("#2b6ea6", "#9ccbf2"),
        "mode_2": ("#8a3a5c", "#f2a7c4"),
        "bath_ic": ("#2d6b3e", "#8fd9a0"),
        "work": ("#4a4594", "#b9b3f2"),
        "carry": ("#4a4594", "#b9b3f2"),
        "pt": ("#1e6b38", "#8fd9a0"),
        "glow": "#ffd166",
        "debug": "#9e9e9e",
    },
}

TIMING = {
    "blank_hold": 0.15,
    "ic_draw": 0.5,
    "column_draw": 0.6,
    "camera_move": 0.75,
    "emphasis": 0.2,
    "retract": 0.3,
    "merge": 0.65,
    "split": 0.65,
    "carry_advance": 0.55,
    "bend": 1.0,
    "marker_draw": 0.35,
    "reframe": 0.7,
    "beat_gap": 0.1,
    "result_hold": 1.8,
    "loop_fade": 0.4,
}

LAYOUT = {
    "frame_height": 4.5,
    "n_steps": 3,
    "pitch": 2.0,
    "x_right": 2.0,
    "row_y": {"E1": 0.7, "E2": -0.7},
    "core_side": 0.66,
    "corner": 0.09,
    "ic_gap": 1.3,
    "ic_side": 0.56,
    "stub_len": 0.52,
    "terminal_len": 0.55,
    "work_size": (0.8, 1.6),
    "carry_size": (0.46, 0.9),
    "carry_split_dx": 0.62,
    "carry_wait_dx": 0.85,
    "socket_drop": 0.66,
    "marker_r": 0.085,
    "camera_padding": 0.35,
}

DEBUG = {
    "show_ids": False,
    "show_port_names": False,
    "show_bounds": False,
    "highlight_moving": False,
    "log_beats": True,
    "save_checkpoints": False,
    "stop_after": None,
}

EXPORT = {
    "final": {"pixel_width": 960, "pixel_height": 540, "fps": 24},
    "draft": {"pixel_width": 480, "pixel_height": 270, "fps": 12},
    "gif_dither": "sierra2_4a",
    "gif_alpha_threshold": 128,
    "mp4_background": {"light": "#ffffff", "dark": "#1f2424"},
    "poster_offset": 0.5,
}

Z = {"wire": 1, "emphasis": 1.5, "closure": 1.75, "body": 2, "marker": 3, "glow": 4, "debug": 5}


def active_theme() -> str:
    theme = os.environ.get("PT_ANIM_THEME", "light").strip().lower()
    if theme not in PALETTES:
        raise ValueError(f"PT_ANIM_THEME must be one of {sorted(PALETTES)}, got {theme!r}")
    return theme


def screen_stroke(px: float, frame_width: float) -> float:
    """Manim stroke width that renders as ``px`` pixels at the reference export width.

    Cairo strokes live in scene units, so the camera width enters explicitly.
    """
    return px * frame_width / (STYLE["reference_pixel_width"] * 0.01)


def set_screen_stroke(mob: VMobject, px: float, frame_width: float) -> VMobject:
    mob.pt_px = px
    mob.set_stroke(width=screen_stroke(px, frame_width))
    return mob


# --------------------------------------------------------------------------
# Small local records
# --------------------------------------------------------------------------


class Status(Enum):
    HIDDEN = "hidden"
    LIVE = "live"
    CONSUMED = "consumed"


class PortKind(Enum):
    PHYSICAL_IN = "physical_in"
    PHYSICAL_OUT = "physical_out"
    VIRTUAL = "virtual"
    OPEN_RESULT = "open_result"


class Route(Enum):
    STRAIGHT = "straight"
    FRONTIER = "frontier"
    DOWN_SOCKET = "down_socket"


@dataclass(frozen=True)
class PortRef:
    owner_id: str
    port_name: str
    semantic_kind: PortKind


@dataclass
class TensorGlyph:
    id: str
    body: VMobject
    kind: str
    port_specs: dict[str, np.ndarray] = field(default_factory=dict)
    markers: dict[str, VMobject] = field(default_factory=dict)
    label: VMobject | None = None
    status: Status = Status.HIDDEN

    def port(self, name: str, centre: np.ndarray | None = None) -> np.ndarray:
        base = self.body.get_center() if centre is None else centre
        return base + self.port_specs[name]


@dataclass
class WireRecord:
    id: str
    a: PortRef
    b: PortRef
    path: VMobject
    route_kind: Route
    status: Status = Status.HIDDEN


@dataclass
class SceneState:
    glyphs: dict[str, TensorGlyph] = field(default_factory=dict)
    wires: dict[str, WireRecord] = field(default_factory=dict)
    fixed: dict[str, np.ndarray] = field(default_factory=dict)
    cell: dict[tuple[str, int], str] = field(default_factory=dict)
    completed: dict[int, str] = field(default_factory=dict)
    carry_id: str | None = None
    working_id: str | None = None
    closure_markers: dict[str, VMobject] = field(default_factory=dict)
    transient_ids: set[str] = field(default_factory=set)
    camera_width: float = 1.0


@dataclass
class Beat:
    name: str
    animations: list
    moving_ids: tuple[str, ...] = ()
    consumed_ids: tuple[str, ...] = ()
    created_ids: tuple[str, ...] = ()
    commit: Callable[[], None] = lambda: None
    prepare: Callable[[MovingCameraScene], None] = lambda scene: None
    finish: Callable[[MovingCameraScene], None] = lambda scene: None
    discard: list[VMobject] = field(default_factory=list)


# --------------------------------------------------------------------------
# Pure geometry factories
# --------------------------------------------------------------------------


def point(x: float, y: float) -> np.ndarray:
    return np.array([x, y, 0.0])


def column_x(k: int) -> float:
    return LAYOUT["x_right"] - k * LAYOUT["pitch"]


def split_cubic(p0, p1, p2, p3) -> np.ndarray:
    """One cubic as two cubic halves (8 control points) so every wire interpolates."""
    a, b, c = (p0 + p1) / 2, (p1 + p2) / 2, (p2 + p3) / 2
    d, e = (a + b) / 2, (b + c) / 2
    m = (d + e) / 2
    return np.array([p0, a, d, m, m, e, c, p3])


def route_points(route: Route, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if route == Route.DOWN_SOCKET:
        # Quarter-ellipse: horizontal tangent on the core, vertical tangent on the
        # marker. Both controls sit in the same bounding box, so the bend is convex.
        kappa = 0.551915024494
        dx, dy = b[0] - a[0], b[1] - a[1]
        return split_cubic(a, a + point(kappa * dx, 0.0), b - point(0.0, kappa * dy), b)
    if route == Route.FRONTIER:
        # a: bath input on the next column (enters from the east); b: carry's top/bottom virtual port.
        dx, dy = b[0] - a[0], a[1] - b[1]
        return split_cubic(a, a + RIGHT * max(0.12, 0.55 * dx), b + UP * np.sign(dy) * max(0.08, 0.8 * abs(dy)), b)
    return split_cubic(a, a + (b - a) / 3, a + 2 * (b - a) / 3, b)


def styled_body(mob: VMobject, colors: tuple[str, str], width: float) -> VMobject:
    mob.set_fill(colors[0], opacity=1.0)
    mob.set_stroke(colors[1])
    set_screen_stroke(mob, STYLE["outline_px"], width)
    mob.set_z_index(Z["body"])
    return mob


def make_box(center: np.ndarray, size: tuple[float, float], colors: tuple[str, str], width: float) -> VMobject:
    box = RoundedRectangle(width=size[0], height=size[1], corner_radius=LAYOUT["corner"])
    box.move_to(center)
    return styled_body(box, colors, width)


def make_right_triangle(center: np.ndarray, side: float, colors: tuple[str, str], width: float) -> VMobject:
    """Apex on the right, base on the left, so the connecting leg meets the base."""
    h = side * np.sqrt(3) / 2
    tri = Polygon(center + point(2 * h / 3, 0), center + point(-h / 3, side / 2), center + point(-h / 3, -side / 2))
    tri.round_corners(radius=0.12 * side)
    return styled_body(tri, colors, width)


def make_wire_path(points: np.ndarray, color: str, width: float) -> VMobject:
    wire = VMobject()
    wire.set_points(points)
    wire.set_fill(opacity=0)
    wire.set_stroke(color)
    set_screen_stroke(wire, STYLE["wire_px"], width)
    wire.set_cap_style(CapStyleType.ROUND)
    wire.set_z_index(Z["wire"])
    return wire


def make_closure_marker(position: np.ndarray, pal: dict) -> VMobject:
    """Filled disc in the leg colour. It sits behind the rectangle that absorbs it."""
    marker = Circle(radius=LAYOUT["marker_r"], stroke_width=0)
    marker.set_fill(pal["wire"], opacity=1.0)
    marker.set_stroke(width=0)
    marker.move_to(position)
    marker.set_z_index(Z["closure"])
    return marker


def side_attachment(role: str) -> np.ndarray:
    """Open-leg port on the left or right edge, three quarters of the way down."""
    side = LAYOUT["core_side"]
    return point(-side / 2 if role == "out" else side / 2, -side / 4)


def open_socket(core_centre: np.ndarray, role: str) -> tuple[np.ndarray, np.ndarray]:
    """Marker centre, and the point where a downward leg meets the marker's top edge.

    Markers sit a quarter-pitch either side of the core, so the pair under one
    core and the pair between neighbouring cores are equally spaced.
    """
    sign = -1.0 if role == "out" else 1.0
    centre = core_centre + point(sign * LAYOUT["pitch"] / 4, -LAYOUT["socket_drop"])
    return centre, centre + UP * LAYOUT["marker_r"]


def make_marker(role: str, filled: bool, position: np.ndarray, pal: dict, width: float) -> VMobject:
    """Input sockets are teal circles, output sockets amber rounded squares, everywhere."""
    r = LAYOUT["marker_r"]
    color = pal["input"] if role == "in" else pal["output"]
    marker = Circle(radius=r) if role == "in" else RoundedRectangle(width=2 * r, height=2 * r, corner_radius=0.35 * r)
    marker.set_stroke(color)
    set_screen_stroke(marker, STYLE["marker_px"], width)
    marker.set_fill(color, opacity=1.0 if filled else 0.0)
    marker.move_to(position)
    marker.set_z_index(Z["marker"])
    return marker


def box_ports(size: tuple[float, float], extra: dict[str, tuple[float, float]] | None = None) -> dict[str, np.ndarray]:
    w, h = size
    ports = {"n": point(0, h / 2), "s": point(0, -h / 2), "e": point(w / 2, 0), "w": point(-w / 2, 0)}
    for name, (x, y) in (extra or {}).items():
        ports[name] = point(x, y)
    return ports


def work_ports() -> dict[str, np.ndarray]:
    """Fused column: physical legs top/bottom, row-aligned bath ports on both sides, memory bond east."""
    w, h = LAYOUT["work_size"]
    y1 = LAYOUT["row_y"]["E1"]
    return box_ports(LAYOUT["work_size"], {
        "e1": (w / 2, y1), "e2": (w / 2, -y1), "w1": (-w / 2, y1), "w2": (-w / 2, -y1), "mem_e": (w / 2, 0),
    })


def carry_ports() -> dict[str, np.ndarray]:
    """Transfer factor: top/bottom ports are VIRTUAL frontier bonds, not physical legs."""
    w, h = LAYOUT["carry_size"]
    return {"f1": point(0, h / 2), "f2": point(0, -h / 2), "mem_e": point(w / 2, 0)}


def core_ports() -> dict[str, np.ndarray]:
    s = LAYOUT["core_side"]
    ports = box_ports((s, s), {"mem_w": (-s / 2, 0), "mem_e": (s / 2, 0)})
    ports["out"] = side_attachment("out")
    ports["in"] = side_attachment("in")
    return ports


def mobject_bounds(mobs: list[VMobject]) -> tuple[np.ndarray, np.ndarray]:
    group = VGroup(*mobs)
    return group.get_corner(DOWN + LEFT), group.get_corner(UP + RIGHT)


def fit_frame(mobs: list[VMobject], padding: float) -> tuple[np.ndarray, float]:
    """Centre and width that contain ``mobs`` in both dimensions."""
    lo, hi = mobject_bounds(mobs)
    aspect = config.frame_width / config.frame_height
    width = max(hi[0] - lo[0] + 2 * padding, aspect * (hi[1] - lo[1] + 2 * padding))
    return (lo + hi) / 2, width


# --------------------------------------------------------------------------
# Hidden geometry
# --------------------------------------------------------------------------


def port_position(state: SceneState, ref: PortRef) -> np.ndarray:
    if ref.owner_id == "fixed":
        return state.fixed[ref.port_name]
    return state.glyphs[ref.owner_id].port(ref.port_name)


def wire_points(state: SceneState, wire: WireRecord) -> np.ndarray:
    return route_points(wire.route_kind, port_position(state, wire.a), port_position(state, wire.b))


def register_wire(state: SceneState, wire_id: str, a: PortRef, b: PortRef, route: Route, pal: dict) -> WireRecord:
    record = WireRecord(wire_id, a, b, make_wire_path(np.zeros((8, 3)), pal["wire"], state.camera_width), route)
    record.path.set_points(wire_points(state, record))
    state.wires[wire_id] = record
    return record


def add_glyph(state: SceneState, glyph: TensorGlyph, pal: dict) -> TensorGlyph:
    if glyph.id in state.glyphs:
        raise AssertionError(f"duplicate glyph id {glyph.id}")
    state.glyphs[glyph.id] = glyph
    if DEBUG["show_ids"]:
        glyph.label = Text(glyph.id, font_size=10, color=pal["debug"]).next_to(glyph.body, UP, buff=0.05)
        glyph.label.set_z_index(Z["debug"])
    return glyph


def make_bath_initial_conditions(state: SceneState, pal: dict) -> None:
    """Single-wire triangles: vectorised bath density operators, one per mode."""
    for m in MODES:
        centre = point(column_x(0) + LAYOUT["ic_gap"], LAYOUT["row_y"][m])
        body = make_right_triangle(centre, LAYOUT["ic_side"], pal["bath_ic"], state.camera_width)
        base = point(body.get_left()[0], body.get_center()[1])
        add_glyph(state, TensorGlyph(f"ic:{m}", body, "bath_ic", {"base": base - body.get_center()}), pal)


def make_propagator_grid(state: SceneState, pal: dict) -> None:
    s = LAYOUT["core_side"]
    n = LAYOUT["n_steps"]
    for k in range(n):
        for m in MODES:
            gid = f"prop:{m}:k{k}"
            colors = pal["mode_1"] if m == "E1" else pal["mode_2"]
            body = make_box(point(column_x(k), LAYOUT["row_y"][m]), (s, s), colors, state.camera_width)
            add_glyph(state, TensorGlyph(gid, body, "propagator", box_ports((s, s))), pal)
            state.cell[(m, k)] = gid
    for k in range(n):
        x = column_x(k)
        top = LAYOUT["row_y"]["E1"] + s / 2 + LAYOUT["stub_len"]
        bottom = LAYOUT["row_y"]["E2"] - s / 2 - LAYOUT["stub_len"]
        state.fixed[f"open:out[{k}]"] = point(x, top)
        state.fixed[f"open:in[{k}]"] = point(x, bottom)
        # E1's top leg is out[k] and E2's bottom leg is in[k] through every transformation.
        register_wire(state, f"phys:out[{k}]", PortRef(f"prop:E1:k{k}", "n", PortKind.PHYSICAL_OUT),
                      PortRef("fixed", f"open:out[{k}]", PortKind.PHYSICAL_OUT), Route.STRAIGHT, pal)
        register_wire(state, f"phys:in[{k}]", PortRef(f"prop:E2:k{k}", "s", PortKind.PHYSICAL_IN),
                      PortRef("fixed", f"open:in[{k}]", PortKind.PHYSICAL_IN), Route.STRAIGHT, pal)
        register_wire(state, f"vert:k{k}", PortRef(f"prop:E1:k{k}", "s", PortKind.VIRTUAL),
                      PortRef(f"prop:E2:k{k}", "n", PortKind.VIRTUAL), Route.STRAIGHT, pal)
        for m in MODES:
            east = PortRef(f"prop:{m}:k{k}", "e", PortKind.VIRTUAL)
            far = PortRef(f"ic:{m}", "base", PortKind.VIRTUAL) if k == 0 else PortRef(f"prop:{m}:k{k - 1}", "w", PortKind.VIRTUAL)
            register_wire(state, f"bath:{m}:k{k}", east, far, Route.STRAIGHT, pal)
    for m in MODES:
        # The closure sits on the wire from the moment the last propagator exists.
        centre = point(column_x(n - 1) - s / 2 - LAYOUT["terminal_len"], LAYOUT["row_y"][m])
        state.fixed[f"terminal:{m}"] = centre + RIGHT * LAYOUT["marker_r"]
        state.closure_markers[m] = make_closure_marker(centre, pal)
        register_wire(state, f"term:{m}", PortRef(f"prop:{m}:k{n - 1}", "w", PortKind.VIRTUAL),
                      PortRef("fixed", f"terminal:{m}", PortKind.VIRTUAL), Route.STRAIGHT, pal)


def ids_revealed_at(k: int) -> tuple[list[str], list[str]]:
    glyphs = [f"prop:{m}:k{k}" for m in MODES]
    wires = [f"phys:out[{k}]", f"phys:in[{k}]", f"vert:k{k}"] + [f"bath:{m}:k{k}" for m in MODES]
    if k == LAYOUT["n_steps"] - 1:
        wires += [f"term:{m}" for m in MODES]
    return glyphs, wires


# --------------------------------------------------------------------------
# Animation helpers (never call scene.play)
# --------------------------------------------------------------------------


def retract(mob: VMobject, toward: str, run_time: float) -> UpdateFromAlphaFunc:
    """Consumed connection writes out towards ``start``, ``end`` or its ``middle``."""
    original = mob.copy()

    def update(m: VMobject, alpha: float) -> None:
        a = float(np.clip(alpha, 0.0, 1.0))
        lo, hi = {"start": (0.0, 1.0 - a), "end": (a, 1.0), "middle": (a / 2, 1.0 - a / 2)}[toward]
        m.pointwise_become_partial(original, lo, max(hi, lo))
        m.set_stroke(opacity=0.0 if a > 0.92 else 1.0)

    return UpdateFromAlphaFunc(mob, update, run_time=run_time, rate_func=smooth)


def glide_into(mob: VMobject, destination: np.ndarray, run_time: float) -> UpdateFromAlphaFunc:
    """Carry a closure marker onto the body edge, then let it sink into that edge."""
    original = mob.copy()
    origin = original.get_center()

    def update(m: VMobject, alpha: float) -> None:
        a = float(np.clip(alpha, 0.0, 1.0))
        centre = origin + (destination - origin) * a
        scale = 1.0 if a < 0.78 else max((1.0 - a) / 0.22, 1e-4)
        moved = original.copy()
        moved.scale(scale, about_point=origin)
        moved.move_to(centre)
        m.become(moved)
        if a > 0.92:
            m.set_fill(opacity=0.0)
            m.set_stroke(opacity=0.0)

    return UpdateFromAlphaFunc(mob, update, run_time=run_time, rate_func=smooth)


def interpolate_path(mob: VMobject, start: np.ndarray, end: np.ndarray, run_time: float) -> UpdateFromAlphaFunc:
    def update(m: VMobject, alpha: float) -> None:
        m.set_points(start + (end - start) * alpha)

    return UpdateFromAlphaFunc(mob, update, run_time=run_time, rate_func=smooth)


def emphasis_overlay(path: VMobject, pal: dict, width: float) -> VMobject:
    overlay = path.copy()
    overlay.set_stroke(pal["glow"], opacity=1.0)
    set_screen_stroke(overlay, STYLE["emphasis_px"], width)
    overlay.set_z_index(Z["emphasis"])
    return overlay


def survivor_animations(state: SceneState, port_map: dict[str, tuple[str, PortRef]],
                        new_ports: dict[str, np.ndarray], run_time: float,
                        new_routes: dict[str, Route] | None = None) -> list:
    """Interpolate surviving wires onto ports given by ``new_ports[owner_id:port]``."""
    anims = []
    for wid, (end, ref) in port_map.items():
        wire = state.wires[wid]
        a, b = (ref, wire.b) if end == "a" else (wire.a, ref)
        resolve = lambda r: new_ports.get(f"{r.owner_id}:{r.port_name}", None)
        pa = resolve(a) if resolve(a) is not None else port_position(state, a)
        pb = resolve(b) if resolve(b) is not None else port_position(state, b)
        route = (new_routes or {}).get(wid, wire.route_kind)
        anims.append(interpolate_path(wire.path, wire.path.points.copy(), route_points(route, pa, pb), run_time))
    return anims


def apply_port_map(state: SceneState, port_map: dict[str, tuple[str, PortRef]], new_routes: dict[str, Route] | None = None) -> None:
    for wid, (end, ref) in port_map.items():
        wire = state.wires[wid]
        if end == "a":
            wire.a = ref
        else:
            wire.b = ref
        if new_routes and wid in new_routes:
            wire.route_kind = new_routes[wid]


def glyph_port_lookup(glyph: TensorGlyph, centre: np.ndarray | None = None) -> dict[str, np.ndarray]:
    return {f"{glyph.id}:{name}": glyph.port(name, centre) for name in glyph.port_specs}


# --------------------------------------------------------------------------
# Choreography: each plan_* returns a Beat
# --------------------------------------------------------------------------


def plan_draw_initial_conditions(state: SceneState) -> Beat:
    glyphs = [state.glyphs[f"ic:{m}"] for m in MODES]

    def commit() -> None:
        for g in glyphs:
            g.status = Status.LIVE

    return Beat("draw_initial_conditions", [DrawBorderThenFill(g.body, run_time=TIMING["ic_draw"]) for g in glyphs],
                created_ids=tuple(g.id for g in glyphs), commit=commit)


def plan_reveal_column(scene: "ACECompressionScene", k: int, visible: list[VMobject]) -> Beat:
    """Draw column k, join its bath wires to the region on the right, and widen the camera."""
    state = scene.state
    glyph_ids, wire_ids = ids_revealed_at(k)
    drawn = [state.glyphs[g].body for g in glyph_ids] + [state.wires[w].path for w in wire_ids]
    closures = list(state.closure_markers.values()) if k == LAYOUT["n_steps"] - 1 else []
    new_mobs = drawn + closures
    centre, width = fit_frame(visible + new_mobs, LAYOUT["camera_padding"])
    old_mobs = list(visible)
    for mob in drawn:
        set_screen_stroke(mob, mob.pt_px, width)
    frame = scene.camera.frame
    camera = frame.animate(run_time=TIMING["camera_move"], rate_func=smooth).move_to(centre).set(width=width)

    def compensate(_, alpha: float) -> None:
        current = frame.width
        for mob in old_mobs:
            mob.set_stroke(width=screen_stroke(mob.pt_px, current))

    holder = VMobject()
    draw = [DrawBorderThenFill(state.glyphs[g].body, run_time=TIMING["column_draw"]) for g in glyph_ids]
    draw += [Create(state.wires[w].path, run_time=TIMING["column_draw"]) for w in wire_ids]
    draw += [GrowFromCenter(m, run_time=TIMING["column_draw"]) for m in closures]

    def commit() -> None:
        state.camera_width = width
        for g in glyph_ids:
            state.glyphs[g].status = Status.LIVE
        for w in wire_ids:
            state.wires[w].status = Status.LIVE

    return Beat(
        f"reveal_column_k{k}",
        [camera, UpdateFromAlphaFunc(holder, compensate, run_time=TIMING["camera_move"]), AnimationGroup(*draw)],
        created_ids=tuple(glyph_ids) + tuple(wire_ids),
        commit=commit,
        discard=[holder, frame],
    )


def plan_merge(
    state: SceneState,
    name: str,
    source_ids: list[str],
    consumed_wire_ids: list[tuple[str, str]],
    target: TensorGlyph,
    port_map: dict[str, tuple[str, PortRef]],
    pal: dict,
    on_commit: Callable[[], None] | None = None,
    vanish: list[tuple[VMobject, np.ndarray]] | None = None,
) -> Beat:
    """Local contraction: emphasise and write out internal edges, then morph the bodies.

    Surviving wires in ``port_map`` are re-attached to ``target`` ports during
    the morph. Each ``vanish`` marker glides to the body point it is absorbed by.
    """
    width = state.camera_width
    consumed_paths = [(state.wires[w].path, toward) for w, toward in consumed_wire_ids]
    overlays = [emphasis_overlay(p, pal, width) for p, _ in consumed_paths]
    emphasis = AnimationGroup(*[Create(o, run_time=TIMING["emphasis"]) for o in overlays])
    write_out = AnimationGroup(
        *[retract(p, t, TIMING["retract"]) for p, t in consumed_paths],
        *[retract(o, t, TIMING["retract"]) for o, (_, t) in zip(overlays, consumed_paths)],
        *[glide_into(m, dest, TIMING["retract"]) for m, dest in (vanish or [])],
    )
    # One coincident target copy per source body so every body visibly slides and
    # morphs onto the target; the copies are swapped for the real target after play.
    composite = VGroup(*[state.glyphs[g].body for g in source_ids])
    target_copies = VGroup(*[target.body.copy() for _ in source_ids])
    survivors = survivor_animations(state, port_map, glyph_port_lookup(target), TIMING["merge"])
    morph = AnimationGroup(ReplacementTransform(composite, target_copies, run_time=TIMING["merge"]), *survivors)
    steps = [emphasis, write_out, morph] if consumed_paths else [morph]

    def prepare(scene) -> None:
        scene.remove(*composite.submobjects)
        scene.add(composite)

    def finish(scene) -> None:
        scene.remove(target_copies, composite)
        scene.add(target.body)

    def commit() -> None:
        for gid in source_ids:
            state.glyphs[gid].status = Status.CONSUMED
        for wid, _ in consumed_wire_ids:
            state.wires[wid].status = Status.CONSUMED
        apply_port_map(state, port_map)
        target.status = Status.LIVE
        add_glyph(state, target, pal)
        if on_commit is not None:
            on_commit()

    return Beat(
        name,
        [Succession(*steps)],
        moving_ids=tuple(source_ids) + tuple(port_map),
        consumed_ids=tuple(source_ids) + tuple(w for w, _ in consumed_wire_ids),
        created_ids=(target.id,),
        commit=commit,
        prepare=prepare,
        finish=finish,
        discard=[p for p, _ in consumed_paths] + overlays + [m for m, _ in (vanish or [])] + [composite],
    )


def incident_port_map(state: SceneState, old_id: str, mapping: dict[str, str], target_id: str) -> dict[str, tuple[str, PortRef]]:
    """Re-attach every live wire on ``old_id`` port ``p`` to ``target_id`` port ``mapping[p]``."""
    out = {}
    for wire in state.wires.values():
        if wire.status != Status.LIVE:
            continue
        for end in ("a", "b"):
            ref = getattr(wire, end)
            if ref.owner_id == old_id and ref.port_name in mapping:
                out[wire.id] = (end, PortRef(target_id, mapping[ref.port_name], ref.semantic_kind))
    return out


def plan_absorb_initial_condition(state: SceneState, mode: str, pal: dict) -> Beat:
    old = state.cell[(mode, 0)]
    s = LAYOUT["core_side"]
    colors = pal["mode_1"] if mode == "E1" else pal["mode_2"]
    target = TensorGlyph(f"{old}:ic", make_box(state.glyphs[old].body.get_center(), (s, s), colors, state.camera_width), "propagator", box_ports((s, s)))
    port_map = incident_port_map(state, old, {p: p for p in "nsew"}, target.id)
    port_map.pop(f"bath:{mode}:k0", None)

    def on_commit() -> None:
        state.cell[(mode, 0)] = target.id

    return plan_merge(state, f"absorb_ic:{mode}", [f"ic:{mode}", old], [(f"bath:{mode}:k0", "start")], target, port_map, pal, on_commit)


def plan_join_column(state: SceneState, k: int, pal: dict) -> Beat:
    """Two stacked propagators and their vertical bond fuse into one working rectangle."""
    top, bottom = state.cell[("E1", k)], state.cell[("E2", k)]
    target = TensorGlyph(f"work:k{k}", make_box(point(column_x(k), 0), LAYOUT["work_size"], pal["work"], state.camera_width), "work", work_ports())
    port_map = incident_port_map(state, top, {"n": "n", "e": "e1", "w": "w1"}, target.id)
    port_map.update(incident_port_map(state, bottom, {"s": "s", "e": "e2", "w": "w2"}, target.id))

    def on_commit() -> None:
        state.working_id = target.id

    return plan_merge(state, f"join_column_k{k}", [top, bottom], [(f"vert:k{k}", "middle")], target, port_map, pal, on_commit)


def plan_absorb_carry(state: SceneState, k: int, pal: dict) -> Beat:
    """Carry + fused column: the frontier bath connections between them close."""
    work, carry = state.working_id, state.carry_id
    target = TensorGlyph(f"work:k{k}:carry", make_box(point(column_x(k), 0), LAYOUT["work_size"], pal["work"], state.camera_width), "work", work_ports())
    port_map = incident_port_map(state, work, {p: p for p in ("n", "s", "w1", "w2")}, target.id)
    port_map.update(incident_port_map(state, carry, {"mem_e": "mem_e"}, target.id))
    consumed = [(f"bath:{m}:k{k}", "middle") for m in MODES]

    def on_commit() -> None:
        state.working_id = target.id
        state.carry_id = None

    return plan_merge(state, f"absorb_carry_k{k}", [carry, work], consumed, target, port_map, pal, on_commit)


def plan_split_completed_core(state: SceneState, k: int, pal: dict) -> Beat:
    """Working rectangle -> completed core (stays at k, keeps the physical legs) + carry (moves on).

    The completed core stays on the RIGHT of the carry moving LEFT; canonicality is not drawn.
    """
    work = state.glyphs[state.working_id]
    width = state.camera_width
    core = TensorGlyph(f"core:k{k}", make_box(point(column_x(k), 0), (LAYOUT["core_side"],) * 2, pal["pt"], width), "pt_core", core_ports())
    carry_centre = point(column_x(k) - LAYOUT["carry_split_dx"], 0)
    carry = TensorGlyph(f"carry:k{k}", make_box(carry_centre, LAYOUT["carry_size"], pal["carry"], width), "carry", carry_ports())

    port_map = incident_port_map(state, work.id, {"n": "n", "s": "s", "mem_e": "mem_e"}, core.id)
    port_map.update(incident_port_map(state, work.id, {"w1": "f1", "w2": "f2"}, carry.id))
    new_routes = {wid: Route.FRONTIER for wid, (_, ref) in port_map.items() if ref.owner_id == carry.id}
    lookup = {**glyph_port_lookup(core), **glyph_port_lookup(carry)}
    survivors = survivor_animations(state, port_map, lookup, TIMING["split"], new_routes)

    seam = point(column_x(k) - LAYOUT["work_size"][0] / 2, 0)
    bond = WireRecord(
        f"mem:k{k}", PortRef(core.id, "mem_w", PortKind.VIRTUAL), PortRef(carry.id, "mem_e", PortKind.VIRTUAL),
        make_wire_path(route_points(Route.STRAIGHT, seam, seam), pal["wire"], width), Route.STRAIGHT,
    )
    bond_end = route_points(Route.STRAIGHT, core.port("mem_w"), carry.port("mem_e"))
    work_copy = work.body.copy()
    split = AnimationGroup(
        ReplacementTransform(work.body, core.body, run_time=TIMING["split"]),
        ReplacementTransform(work_copy, carry.body, run_time=TIMING["split"]),
        interpolate_path(bond.path, bond.path.points.copy(), bond_end, TIMING["split"]),
        *survivors,
    )
    def prepare(scene) -> None:
        scene.add(work_copy, bond.path)

    def commit() -> None:
        work.status = Status.CONSUMED
        apply_port_map(state, port_map, new_routes)
        for glyph in (core, carry):
            glyph.status = Status.LIVE
            add_glyph(state, glyph, pal)
        bond.status = Status.LIVE
        state.wires[bond.id] = bond
        state.completed[k] = core.id
        state.carry_id = carry.id
        state.working_id = None

    return Beat(
        f"split_completed_core_k{k}", [split], moving_ids=(work.id,) + tuple(port_map),
        consumed_ids=(work.id,), created_ids=(core.id, carry.id, bond.id), commit=commit, prepare=prepare,
    )


def plan_advance_carry(state: SceneState, next_k: int) -> Beat:
    """Only the carry and its incident routes move; the completed core keeps its legs."""
    carry = state.glyphs[state.carry_id]
    new_centre = point(column_x(next_k) + LAYOUT["carry_wait_dx"], 0)
    port_map = incident_port_map(state, carry.id, {p: p for p in carry.port_specs}, carry.id)
    survivors = survivor_animations(state, port_map, glyph_port_lookup(carry, new_centre), TIMING["carry_advance"])
    move = carry.body.animate(run_time=TIMING["carry_advance"], rate_func=smooth).move_to(new_centre)
    return Beat(f"advance_carry_to_k{next_k}", [move, *survivors], moving_ids=(carry.id,) + tuple(port_map))


def plan_trace_final_bath_boundary(state: SceneState, pal: dict) -> Beat:
    """Bath boundary indices, already capped, are traced out. The system leg out[2] stays."""
    k = LAYOUT["n_steps"] - 1
    work = state.working_id
    core = TensorGlyph(f"core:k{k}", make_box(point(column_x(k), 0), (LAYOUT["core_side"],) * 2, pal["pt"], state.camera_width), "pt_core", core_ports())
    port_map = incident_port_map(state, work, {"n": "n", "s": "s", "mem_e": "mem_e"}, core.id)
    consumed = [(f"term:{m}", "start") for m in MODES]
    if any(w.startswith("phys") for w, _ in consumed):
        raise AssertionError("trace must not consume a system leg")
    # The cap rides into the rectangle's west edge instead of shrinking where it sits.
    markers = [
        (state.closure_markers[m], state.wires[f"term:{m}"].path.points[0].copy() + RIGHT * 0.22)
        for m in MODES
    ]

    def on_commit() -> None:
        state.completed[k] = core.id
        state.working_id = None
        state.closure_markers.clear()

    return plan_merge(state, "trace_final_bath_boundary", [work], consumed, core, port_map, pal, on_commit, markers)


def bend_leg_animation(wire: WireRecord, core_centre: np.ndarray, role: str, run_time: float) -> UpdateFromAlphaFunc:
    """Attachment slides to the side edge while the open end swings down to the marker boundary."""
    half = LAYOUT["core_side"] / 2
    attach = core_centre + side_attachment(role)
    _, end_final = open_socket(core_centre, role)
    start_attach = wire.path.points[0] - core_centre
    start_end = wire.path.points[-1] - core_centre
    th0 = np.arctan2(start_attach[1], start_attach[0])
    th1 = np.arctan2((attach - core_centre)[1], (attach - core_centre)[0])
    phi0 = np.arctan2(start_end[1], start_end[0])
    phi1 = np.arctan2((end_final - core_centre)[1], (end_final - core_centre)[0])
    if role == "out":
        if th1 < th0:
            th1 += 2 * np.pi
        if phi1 < phi0:
            phi1 += 2 * np.pi
    r0, r1 = np.linalg.norm(start_end), np.linalg.norm(end_final - core_centre)

    def update(m: VMobject, alpha: float) -> None:
        if alpha >= 1.0:
            m.set_points(route_points(Route.DOWN_SOCKET, attach, end_final))
            return
        th = th0 + (th1 - th0) * alpha
        direction = point(np.cos(th), np.sin(th))
        p = core_centre + direction * half / max(abs(direction[0]), abs(direction[1]))
        phi = phi0 + (phi1 - phi0) * alpha
        e = core_centre + point(np.cos(phi), np.sin(phi)) * (r0 + (r1 - r0) * alpha)
        straight = e - p
        chord = split_cubic(p, p + straight / 3, e - straight / 3, e)
        arc = route_points(Route.DOWN_SOCKET, p, e)
        m.set_points(chord + (arc - chord) * alpha)

    return UpdateFromAlphaFunc(wire.path, update, run_time=run_time, rate_func=smooth)


def plan_bend_all_physical_legs(state: SceneState, pal: dict) -> Beat:
    """Diagram rerouting only: port IDs and ordering are preserved."""
    bends, markers, port_map = [], [], {}
    for k, core_id in state.completed.items():
        core = state.glyphs[core_id]
        centre = core.body.get_center()
        for role in ("out", "in"):
            wire = state.wires[f"phys:{role}[{k}]"]
            bends.append(bend_leg_animation(wire, centre, role, TIMING["bend"]))
            socket = f"socket:{role}[{k}]"
            marker_centre, boundary = open_socket(centre, role)
            state.fixed[socket] = boundary
            marker = make_marker(role, False, marker_centre, pal, state.camera_width)
            core.markers[role] = marker
            markers.append(GrowFromCenter(marker, run_time=TIMING["marker_draw"]))
            port_map[wire.id] = (wire.a, wire.b, socket, role)

    def commit() -> None:
        for wid, (a, b, socket, role) in port_map.items():
            wire = state.wires[wid]
            wire.a = PortRef(a.owner_id, role, a.semantic_kind)
            wire.b = PortRef("fixed", socket, b.semantic_kind)
            wire.route_kind = Route.DOWN_SOCKET
            if np.linalg.norm(wire.path.points[-1] - state.fixed[socket]) > 1e-6:
                raise AssertionError(f"{wid} does not end on its socket")

    return Beat("bend_all_physical_legs", [Succession(AnimationGroup(*bends), AnimationGroup(*markers))],
                moving_ids=tuple(port_map), commit=commit)


def plan_final_reframe(scene: "ACECompressionScene") -> Beat:
    state = scene.state
    live = scene.live_mobjects()
    centre, width = fit_frame(live, LAYOUT["camera_padding"])
    frame = scene.camera.frame

    def compensate(_, alpha: float) -> None:
        for mob in live:
            mob.set_stroke(width=screen_stroke(mob.pt_px, frame.width))

    holder = VMobject()

    def commit() -> None:
        state.camera_width = width

    return Beat(
        "final_reframe",
        [frame.animate(run_time=TIMING["reframe"], rate_func=smooth).move_to(centre).set(width=width),
         UpdateFromAlphaFunc(holder, compensate, run_time=TIMING["reframe"])],
        commit=commit, discard=[holder, frame],
    )


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------


class ACECompressionScene(MovingCameraScene):
    def setup(self) -> None:
        super().setup()
        self.theme = active_theme()
        self.pal = PALETTES[self.theme]
        self.state = SceneState()
        self.poster_time: float | None = None

    # -- lifecycle --------------------------------------------------------

    def play_beat(self, beat: Beat) -> None:
        self.play_parallel_beats(beat)

    def play_parallel_beats(self, *beats: Beat) -> None:
        moving = {i for beat in beats for i in (*beat.moving_ids, *beat.consumed_ids)}
        fixed = {gid: g.body.get_center().copy() for gid, g in self.state.glyphs.items()
                 if g.status == Status.LIVE and gid not in moving}
        for beat in beats:
            beat.prepare(self)
        anims = [a for beat in beats for a in beat.animations]
        if anims:
            self.play(*anims)
        for beat in beats:
            beat.finish(self)
            if beat.discard:
                self.remove(*beat.discard)
            for mob in beat.discard:
                mob.clear_updaters()
            beat.commit()
            if DEBUG["log_beats"]:
                print(f"[beat] {beat.name}: moving={list(beat.moving_ids)} consumed={list(beat.consumed_ids)} created={list(beat.created_ids)}")
        for gid, centre in fixed.items():
            if np.linalg.norm(self.state.glyphs[gid].body.get_center() - centre) > 1e-6:
                raise AssertionError(f"untouched glyph {gid} moved during {[b.name for b in beats]}")
        self.assert_invariants()

    def checkpoint(self, name: str) -> None:
        if DEBUG["save_checkpoints"]:
            out = Path(config.media_dir) / "checkpoints"
            out.mkdir(parents=True, exist_ok=True)
            self.renderer.update_frame(self)
            self.renderer.get_image().save(out / f"{SCENE_STEM}-{self.theme}-{name}.png")
        if DEBUG["log_beats"]:
            print(f"[checkpoint] {name}")
        if DEBUG["stop_after"] == name:
            raise EndSceneEarlyException()

    def live_mobjects(self) -> list[VMobject]:
        state = self.state
        mobs = []
        for g in state.glyphs.values():
            if g.status == Status.LIVE:
                mobs += [g.body, *g.markers.values()]
                if g.label is not None:
                    mobs.append(g.label)
        mobs += [w.path for w in state.wires.values() if w.status == Status.LIVE]
        for mode, marker in state.closure_markers.items():
            if state.wires[f"term:{mode}"].status == Status.LIVE:
                mobs.append(marker)
        return mobs

    def assert_invariants(self) -> None:
        state = self.state
        live = {id(m) for m in self.get_mobject_family_members()}
        for glyph in state.glyphs.values():
            if glyph.status == Status.LIVE and id(glyph.body) not in live:
                raise AssertionError(f"live glyph {glyph.id} missing from scene")
            if glyph.status == Status.CONSUMED and id(glyph.body) in live:
                raise AssertionError(f"consumed glyph {glyph.id} still drawn")
        for wire in state.wires.values():
            if wire.status == Status.LIVE:
                for ref in (wire.a, wire.b):
                    if ref.owner_id != "fixed" and state.glyphs[ref.owner_id].status != Status.LIVE:
                        raise AssertionError(f"wire {wire.id} references non-live {ref.owner_id}")
                if np.linalg.norm(wire.path.points[0] - port_position(state, wire.a)) > 1e-6:
                    raise AssertionError(f"wire {wire.id} detached from {wire.a.owner_id}:{wire.a.port_name}")
                if np.linalg.norm(wire.path.points[-1] - port_position(state, wire.b)) > 1e-6:
                    raise AssertionError(f"wire {wire.id} detached from {wire.b.owner_id}:{wire.b.port_name}")
            elif wire.status == Status.CONSUMED and id(wire.path) in live:
                raise AssertionError(f"consumed wire {wire.id} still drawn")
        expected = {id(m) for mob in self.live_mobjects() for m in mob.get_family()}
        ghosts = [m for m in self.get_mobject_family_members() if m.has_points() and id(m) not in expected]
        if ghosts:
            raise AssertionError(f"{len(ghosts)} unowned mobjects left in the scene: {ghosts[:3]}")

    def assert_final_pt(self) -> None:
        state = self.state
        live = [g for g in state.glyphs.values() if g.status == Status.LIVE]
        wires = [w for w in state.wires.values() if w.status == Status.LIVE]
        cores = [g for g in live if g.kind == "pt_core"]
        bonds = [w for w in wires if w.id.startswith("mem:")]
        phys = [w for w in wires if w.id.startswith("phys:")]
        if (len(cores), len(live), len(bonds), len(phys), len(wires)) != (3, 3, 2, 6, 8):
            raise AssertionError(f"final PT has {len(cores)} cores, {len(live)} glyphs, {len(bonds)} bonds, {len(phys)} legs, {len(wires)} wires")
        if any(w.id.startswith(("term:", "bath:")) for w in wires):
            raise AssertionError("bath terminals survived into the final PT")

    def mark_poster(self) -> None:
        self.poster_time = self.renderer.time + EXPORT["poster_offset"]

    def reset_loop(self) -> None:
        mobs = list(self.mobjects)
        if mobs:
            self.play(*[FadeOut(m, scale=0.96) for m in mobs], run_time=TIMING["loop_fade"])
        for m in mobs:
            m.clear_updaters()
        self.clear()
        if self.mobjects:
            raise AssertionError("reset_loop left mobjects in the scene")
        self.camera.frame.move_to(self.opening_centre).set(width=self.opening_width)
        self.wait(TIMING["blank_hold"])

    def pause(self) -> None:
        self.wait(TIMING["beat_gap"])

    # -- story ------------------------------------------------------------

    def construct(self) -> None:
        state, pal = self.state, self.pal
        make_bath_initial_conditions(state, pal)
        ics = [state.glyphs[f"ic:{m}"].body for m in MODES]
        self.opening_centre, self.opening_width = fit_frame(ics, LAYOUT["camera_padding"])
        self.camera.frame.move_to(self.opening_centre).set(width=self.opening_width)
        state.camera_width = self.opening_width
        for mob in ics:
            set_screen_stroke(mob, mob.pt_px, self.opening_width)
        make_propagator_grid(state, pal)
        self.wait(TIMING["blank_hold"])

        self.play_beat(plan_draw_initial_conditions(state))
        visible = list(ics)
        for k in range(LAYOUT["n_steps"]):
            self.play_beat(plan_reveal_column(self, k, visible))
            visible = self.live_mobjects()
        self.checkpoint("ace_grid")
        self.pause()

        self.play_parallel_beats(*[plan_absorb_initial_condition(state, m, pal) for m in MODES])
        for k in range(LAYOUT["n_steps"]):
            self.play_beat(plan_join_column(state, k, pal))
            if state.carry_id is not None:
                self.play_beat(plan_absorb_carry(state, k, pal))
            if k < LAYOUT["n_steps"] - 1:
                self.play_beat(plan_split_completed_core(state, k, pal))
                self.play_beat(plan_advance_carry(state, k + 1))
            else:
                self.play_beat(plan_trace_final_bath_boundary(state, pal))
            self.checkpoint(f"ace_after_k{k}")

        self.assert_final_pt()
        self.play_beat(plan_bend_all_physical_legs(state, pal))
        self.play_beat(plan_final_reframe(self))
        self.assert_final_pt()
        self.checkpoint("ace_final_pt")

        self.mark_poster()
        self.wait(TIMING["result_hold"])
        self.reset_loop()


# --------------------------------------------------------------------------
# Render entry point
# --------------------------------------------------------------------------


def vertical_content_window(movie: Path, width: int, height: int) -> tuple[int, int]:
    """Return the top row and height of a crop that keeps every opaque pixel.

    Rows that stay empty for the whole clip are dropped. A 22px margin at a
    540px-tall frame (scaled with height) is kept, and the window is expanded
    outward to an even height. The horizontal extent is never changed.
    """
    raw = subprocess.check_output(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(movie),
            "-vf", "format=gbrap,unpremultiply=inplace=1:planes=7,alphaextract,format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ]
    )
    frame = width * height
    n = len(raw) // frame
    if n == 0:
        return 0, height
    alpha = np.frombuffer(raw[: n * frame], dtype=np.uint8).reshape(n, height, width)
    rows = np.where((alpha > 8).any(axis=(0, 2)))[0]
    if rows.size == 0:
        return 0, height
    top, bot = int(rows.min()), int(rows.max())
    pad = max(8, round(22 * height / 540))
    y0 = top - pad
    y1 = bot + pad
    if (y1 - y0 + 1) % 2:
        if y0 > 0:
            y0 -= 1
        else:
            y1 += 1
    y0 = max(0, y0)
    y1 = min(height - 1, y1)
    if y0 > top or y1 < bot:
        raise AssertionError(f"vertical crop would cut content: y={top}-{bot}, window={y0}-{y1}")
    return y0, y1 - y0 + 1


def encode_outputs(movie: Path, out_dir: Path, stem: str, theme: str, poster_time: float | None, fps: int, size: tuple[int, int], preset: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ff = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    # Cairo renders premultiplied RGBA but Manim writes it as straight alpha;
    # without this, fades and antialiased edges darken.
    unpremultiply = "format=gbrap,unpremultiply=inplace=1:planes=7"
    y0, crop_h = vertical_content_window(movie, size[0], size[1])
    crop = "" if y0 == 0 and crop_h == size[1] else f"crop={size[0]}:{crop_h}:0:{y0},"
    if preset == "final":
        gif_filter = (
            f"{unpremultiply},{crop}fps={fps},split[a][b];[a]palettegen=reserve_transparent=1:stats_mode=full[p];"
            f"[b][p]paletteuse=dither={EXPORT['gif_dither']}:alpha_threshold={EXPORT['gif_alpha_threshold']}"
        )
        subprocess.run([*ff, "-i", str(movie), "-vf", gif_filter, "-loop", "0", str(out_dir / f"{stem}.gif")], check=True)
    else:
        background = EXPORT["mp4_background"][theme]
        subprocess.run(
            [*ff, "-f", "lavfi", "-i", f"color=c={background}:s={size[0]}x{crop_h}:r={fps}", "-i", str(movie),
             "-filter_complex", f"[1]{unpremultiply}{',' + crop.rstrip(',') if crop else ''}[fg];[0][fg]overlay=shortest=1,format=yuv420p", "-c:v", "libx264",
             "-crf", "18", "-movflags", "+faststart", str(out_dir / f"{stem}.mp4")],
            check=True,
        )
    if poster_time is not None:
        subprocess.run(
            [*ff, "-ss", f"{poster_time:.3f}", "-i", str(movie), "-vf", f"{unpremultiply},{crop}format=rgba",
             "-frames:v", "1", str(out_dir / f"{stem}-poster.png")],
            check=True,
        )


def render_assets(scene_cls: type[MovingCameraScene] = ACECompressionScene) -> None:
    theme = active_theme()
    preset = os.environ.get("PT_ANIM_PRESET", "final").strip().lower()
    spec = EXPORT[preset]
    repo = Path(__file__).resolve().parents[2]
    media = repo / "docs" / "animations" / "media"
    out_dir = repo / "docs" / "src" / "assets" / "animations" if preset == "final" else media / "drafts"
    stem = f"{SCENE_STEM}-{theme}"
    aspect = spec["pixel_width"] / spec["pixel_height"]
    with tempconfig({
        "pixel_width": spec["pixel_width"],
        "pixel_height": spec["pixel_height"],
        "frame_rate": spec["fps"],
        "frame_height": LAYOUT["frame_height"],
        "frame_width": LAYOUT["frame_height"] * aspect,
        "background_opacity": 0.0,
        "format": "mov",
        "media_dir": str(media),
        "output_file": stem,
        "disable_caching": True,
        "verbosity": "WARNING",
        "progress_bar": "none",
    }):
        scene = scene_cls()
        scene.render()
        movie = Path(scene.renderer.file_writer.movie_file_path)
    encode_outputs(movie, out_dir, stem, theme, scene.poster_time, spec["fps"], (spec["pixel_width"], spec["pixel_height"]), preset)
    print(f"[export] {stem}: {out_dir}")


if __name__ == "__main__":
    render_assets()
