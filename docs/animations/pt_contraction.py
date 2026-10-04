# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: docs/animations/pt_contraction.py
# Contributor: Gauthameshwar S.
#
# Generates the schematic two-panel process-tensor contraction animation: one
# panel ends as an open reduced state, the other as a closed scalar.
#
# Run with:
# PT_ANIM_THEME=light docs/animations/.manim_env/bin/python docs/animations/pt_contraction.py
"""Process-tensor contraction animation for the ProcessTensors.jl homepage and README.

Schematic only: no package calls, no numerics. Two identical three-core process
tensors sit side by side; time runs right to left. Both receive the same initial
state and identities. The LEFT panel does not contract its last leg: the remaining
square becomes a right-facing triangle and that leg flattens into a wire pointing
left. The RIGHT panel finishes with an effect and settles into a legless scalar.

Tested with Manim Community v0.21.0 (Cairo renderer), Python 3.12, ffmpeg 8.0.1.

Render from the repository root::

    PT_ANIM_THEME=light docs/animations/.manim_env/bin/python docs/animations/pt_contraction.py
    PT_ANIM_THEME=dark  docs/animations/.manim_env/bin/python docs/animations/pt_contraction.py

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
    Scene,
    Succession,
    Text,
    Uncreate,
    UpdateFromAlphaFunc,
    VGroup,
    VMobject,
    config,
    linear,
    rate_functions,
    smooth,
    tempconfig,
)
from manim.utils.exceptions import EndSceneEarlyException

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

SCENE_STEM = "contraction"

STYLE = {
    "outline_px": 5.0,
    "wire_px": 4.5,
    "marker_px": 4.5,
    "pulse_px": 4.0,
    "emphasis_px": 7.5,
    "reference_pixel_width": 1280,
}

PALETTES = {
    "light": {
        "outline": "#2b2b2b",
        "wire": "#3a3a3a",
        "input": "#138a8a",
        "output": "#c27c0e",
        "pt": ("#d4ead4", "#217a3e"),
        "accumulator": ("#d9c6e8", "#5b2f78"),
        "identity": ("#d5e2f0", "#4a6f96"),
        "state": ("#c8ecea", "#138a8a"),
        "adapter": ("#fdebc8", "#c27c0e"),
        "effect": ("#f7d2c4", "#c0533a"),
        "scalar": ("#fbe7a1", "#a87a00"),
        "glow": "#f2a900",
        "debug": "#7a7a7a",
    },
    "dark": {
        "outline": "#e6e6e6",
        "wire": "#cfcfcf",
        "input": "#4fd1c5",
        "output": "#f2b544",
        "pt": ("#1e6b38", "#8fd9a0"),
        "accumulator": ("#5e3577", "#e9cdf7"),
        "identity": ("#345a80", "#a9c8ea"),
        "state": ("#1f7a76", "#8ff0e6"),
        "adapter": ("#8a5f12", "#f2b544"),
        "effect": ("#a84e36", "#ffb59e"),
        "scalar": ("#9a7414", "#ffe08a"),
        "glow": "#ffd166",
        "debug": "#9e9e9e",
    },
}

TIMING = {
    "blank_hold": 0.15,
    "diagram_draw": 0.6,
    "piece_draw": 0.45,
    "snap_approach": 0.35,
    "snap_settle": 0.12,
    "snap_pulse": 0.28,
    "emphasis": 0.22,
    "retract": 0.32,
    "merge": 0.68,
    "result_settle": 0.75,
    "zoom": 0.85,
    "beat_gap": 0.12,
    "result_hold": 1.8,
    "loop_fade": 0.4,
}

LAYOUT = {
    "frame_height": 4.5,
    "panel_origins": {"left": (-2.55, 0.0), "right": (2.55, 0.0)},
    "n_steps": 3,
    "pitch": 1.55,
    "x_right": 1.55,
    "row_y": 0.55,
    "core_side": 0.42,
    "corner": 0.08,
    "acc_side": 0.5,
    "socket_drop": 0.5,
    "marker_r": 0.075,
    "spawn_drop": 0.48,
    "hover_gap": 0.06,
    "tri_side": 0.4,
    "leg_len": 0.18,
    "open_stub": 0.46,
    "result_gap": 1.15,
    "result_tri_side": 0.52,
    "result_circle_r": 0.26,
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
    "final": {"pixel_width": 1280, "pixel_height": 560, "fps": 24},
    "draft": {"pixel_width": 640, "pixel_height": 280, "fps": 12},
    "gif_dither": "sierra2_4a",
    "gif_alpha_threshold": 128,
    "mp4_background": {"light": "#ffffff", "dark": "#1f2424"},
    "poster_offset": 0.5,
}

Z = {"wire": 1, "emphasis": 1.5, "body": 2, "marker": 3, "glow": 4, "debug": 5}


def active_theme() -> str:
    theme = os.environ.get("PT_ANIM_THEME", "light").strip().lower()
    if theme not in PALETTES:
        raise ValueError(f"PT_ANIM_THEME must be one of {sorted(PALETTES)}, got {theme!r}")
    return theme


def screen_stroke(px: float) -> float:
    """Manim stroke width that renders as ``px`` pixels at the reference export width."""
    return px * config.frame_width / (STYLE["reference_pixel_width"] * 0.01)


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
    DOWN_SOCKET = "down_socket"
    PIECE_LEG = "piece_leg"
    TAIL = "tail"


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

    def port(self, name: str) -> np.ndarray:
        return self.body.get_center() + self.port_specs[name]


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
    panel_id: str
    origin: np.ndarray
    terminal_kind: str
    glyphs: dict[str, TensorGlyph] = field(default_factory=dict)
    wires: dict[str, WireRecord] = field(default_factory=dict)
    sockets: dict[str, VMobject] = field(default_factory=dict)
    socket_centres: dict[str, np.ndarray] = field(default_factory=dict)
    socket_status: dict[str, str] = field(default_factory=dict)
    accumulator_id: str | None = None
    piece_parts: dict[str, list[VMobject]] = field(default_factory=dict)
    transient_ids: set[str] = field(default_factory=set)


@dataclass
class Beat:
    name: str
    animations: list
    moving_ids: tuple[str, ...] = ()
    consumed_ids: tuple[str, ...] = ()
    created_ids: tuple[str, ...] = ()
    commit: Callable[[], None] = lambda: None
    prepare: Callable[[Scene], None] = lambda scene: None
    finish: Callable[[Scene], None] = lambda scene: None
    discard: list[VMobject] = field(default_factory=list)


# --------------------------------------------------------------------------
# Pure geometry factories
# --------------------------------------------------------------------------


def point(x: float, y: float) -> np.ndarray:
    return np.array([x, y, 0.0])


def split_cubic(p0, p1, p2, p3) -> np.ndarray:
    """One cubic as two cubic halves (8 control points) so every wire interpolates."""
    a, b, c = (p0 + p1) / 2, (p1 + p2) / 2, (p2 + p3) / 2
    d, e = (a + b) / 2, (b + c) / 2
    m = (d + e) / 2
    return np.array([p0, a, d, m, m, e, c, p3])


def route_points(route: Route, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if route == Route.DOWN_SOCKET:
        # Quarter-ellipse: horizontal tangent on the core, vertical tangent on the marker.
        kappa = 0.551915024494
        dx, dy = b[0] - a[0], b[1] - a[1]
        return split_cubic(a, a + point(kappa * dx, 0.0), b - point(0.0, kappa * dy), b)
    return split_cubic(a, a + (b - a) / 3, a + 2 * (b - a) / 3, b)


def make_wire_path(points: np.ndarray, color: str) -> VMobject:
    wire = VMobject()
    wire.set_points(points)
    wire.set_fill(opacity=0)
    wire.set_stroke(color, width=screen_stroke(STYLE["wire_px"]))
    wire.set_cap_style(CapStyleType.ROUND)
    wire.set_z_index(Z["wire"])
    return wire


def make_marker(role: str, filled: bool, position: np.ndarray, pal: dict) -> VMobject:
    """Input sockets are teal circles, output sockets amber rounded squares, everywhere."""
    r = LAYOUT["marker_r"]
    color = pal["input"] if role == "in" else pal["output"]
    if role == "in":
        marker = Circle(radius=r)
    else:
        marker = RoundedRectangle(width=2 * r, height=2 * r, corner_radius=0.35 * r)
    marker.set_stroke(color, width=screen_stroke(STYLE["marker_px"]))
    marker.set_fill(color, opacity=1.0 if filled else 0.0)
    marker.move_to(position)
    marker.set_z_index(Z["marker"])
    return marker


def make_box(center: np.ndarray, w: float, h: float, colors: tuple[str, str], corner: float) -> VMobject:
    box = RoundedRectangle(width=w, height=h, corner_radius=corner)
    box.set_fill(colors[0], opacity=1.0)
    box.set_stroke(colors[1], width=screen_stroke(STYLE["outline_px"]))
    box.move_to(center)
    box.set_z_index(Z["body"])
    return box


def make_arrow_body(center: np.ndarray, side: float, colors: tuple[str, str], pointing_right: bool) -> VMobject:
    """Equilateral arrow. The leg meets the vertical base, opposite the apex."""
    h = side * np.sqrt(3) / 2
    sign = 1.0 if pointing_right else -1.0
    tri = Polygon(
        center + point(sign * 2 * h / 3, 0),
        center + point(-sign * h / 3, side / 2),
        center + point(-sign * h / 3, -side / 2),
    )
    tri.round_corners(radius=0.12 * side)
    tri.set_fill(colors[0], opacity=1.0)
    tri.set_stroke(colors[1], width=screen_stroke(STYLE["outline_px"]))
    tri.set_z_index(Z["body"])
    return tri


def make_scalar_circle(center: np.ndarray, colors: tuple[str, str]) -> VMobject:
    circle = Circle(radius=LAYOUT["result_circle_r"])
    circle.set_fill(colors[0], opacity=1.0)
    circle.set_stroke(colors[1], width=screen_stroke(STYLE["outline_px"]))
    circle.move_to(center)
    circle.set_z_index(Z["body"])
    return circle


def debug_label(text: str, position: np.ndarray, pal: dict) -> VMobject:
    label = Text(text, font_size=12, color=pal["debug"])
    label.move_to(position)
    label.set_z_index(Z["debug"])
    return label


# --------------------------------------------------------------------------
# Panel geometry (local coordinates shifted by the panel origin)
# --------------------------------------------------------------------------


def core_x(state: SceneState, k: int) -> float:
    return state.origin[0] + LAYOUT["x_right"] - k * LAYOUT["pitch"]


def row_y(state: SceneState) -> float:
    return state.origin[1] + LAYOUT["row_y"]


def socket_y(state: SceneState) -> float:
    return row_y(state) - LAYOUT["socket_drop"]


def socket_centre(state: SceneState, name: str) -> np.ndarray:
    """Markers sit a quarter-pitch either side of each core, so every gap matches."""
    role, k = name.split("[")[0], int(name[-2])
    sign = 1.0 if role == "in" else -1.0
    return point(core_x(state, k) + sign * LAYOUT["pitch"] / 4, socket_y(state))


def marker_edge(centre: np.ndarray, toward: np.ndarray) -> np.ndarray:
    """Point where a straight horizontal leg, coming from ``toward``, meets the marker."""
    sign = 1.0 if toward[0] >= centre[0] else -1.0
    return point(centre[0] + sign * LAYOUT["marker_r"], centre[1])


def core_port_specs(w: float, h: float) -> dict[str, np.ndarray]:
    """Physical legs leave the side edges three quarters of the way down; memory bonds leave mid-side."""
    return {
        "out": point(-w / 2, -h / 4),
        "in": point(w / 2, -h / 4),
        "mem_w": point(-w / 2, 0),
        "mem_e": point(w / 2, 0),
    }


def port_position(state: SceneState, ref: PortRef) -> np.ndarray:
    if ref.owner_id == "socket":
        return state.socket_centres[ref.port_name]
    return state.glyphs[ref.owner_id].port(ref.port_name)


def leg_end(state: SceneState, ref: PortRef, route: Route) -> np.ndarray:
    pos = port_position(state, ref)
    if route == Route.DOWN_SOCKET and ref.owner_id == "socket":
        return pos + UP * LAYOUT["marker_r"]
    return pos


def wire_points(state: SceneState, wire: WireRecord) -> np.ndarray:
    return route_points(wire.route_kind, leg_end(state, wire.a, wire.route_kind), leg_end(state, wire.b, wire.route_kind))


def register_wire(state: SceneState, wire_id: str, a: PortRef, b: PortRef, route: Route, pal: dict) -> WireRecord:
    record = WireRecord(wire_id, a, b, make_wire_path(np.zeros((8, 3)), pal["wire"]), route)
    record.path.set_points(wire_points(state, record))
    state.wires[wire_id] = record
    return record


def make_pt_panel(panel_id: str, origin: tuple[float, float], terminal_kind: str, pal: dict) -> SceneState:
    """Three cores, two memory bonds, six downward physical legs, six hollow sockets."""
    state = SceneState(panel_id, point(*origin), terminal_kind)
    s = LAYOUT["core_side"]
    for k in range(LAYOUT["n_steps"]):
        gid = f"pt:k{k}"
        body = make_box(point(core_x(state, k), row_y(state)), s, s, pal["pt"], LAYOUT["corner"])
        state.glyphs[gid] = TensorGlyph(gid, body, "pt_core", core_port_specs(s, s))
        if DEBUG["show_ids"]:
            state.glyphs[gid].label = debug_label(f"{panel_id}:{gid}", body.get_top() + UP * 0.15, pal)
        for role in ("out", "in"):
            name = f"{role}[{k}]"
            centre = socket_centre(state, name)
            state.socket_centres[name] = centre
            state.sockets[name] = make_marker(role, False, centre, pal)
            state.socket_status[name] = "empty"
            kind = PortKind.PHYSICAL_OUT if role == "out" else PortKind.PHYSICAL_IN
            register_wire(state, f"leg:{name}", PortRef(gid, role, kind), PortRef("socket", name, kind), Route.DOWN_SOCKET, pal)
    for k in range(LAYOUT["n_steps"] - 1):
        register_wire(
            state, f"mem:{k}",
            PortRef(f"pt:k{k}", "mem_w", PortKind.VIRTUAL), PortRef(f"pt:k{k + 1}", "mem_e", PortKind.VIRTUAL),
            Route.STRAIGHT, pal,
        )
    return state


# --------------------------------------------------------------------------
# Piece factories (built docked; the spawn offset is applied when drawn)
# --------------------------------------------------------------------------


@dataclass
class Piece:
    glyph: TensorGlyph
    legs: dict[str, VMobject]
    endpoints: list[str]
    tail: VMobject | None = None

    def group(self) -> VGroup:
        parts = [self.glyph.body, *self.legs.values(), *self.glyph.markers.values()]
        if self.tail is not None:
            parts.append(self.tail)
        if self.glyph.label is not None:
            parts.append(self.glyph.label)
        return VGroup(*parts)


def make_panel_initial_state(state: SceneState, pal: dict) -> Piece:
    """Preparation: right-pointing arrow to the right of its input, leg running left into the marker."""
    socket = state.socket_centres["in[0]"]
    y = float(socket[1])
    h = LAYOUT["tri_side"] * np.sqrt(3) / 2
    base_x = socket[0] + LAYOUT["marker_r"] + LAYOUT["leg_len"]
    body = make_arrow_body(point(base_x + h / 3, y), LAYOUT["tri_side"], pal["state"], pointing_right=True)
    attach = point(body.get_left()[0], y)
    leg = make_wire_path(route_points(Route.STRAIGHT, attach, marker_edge(socket, attach)), pal["wire"])
    glyph = TensorGlyph("state:init", body, "state", markers={"in[0]": make_marker("in", True, socket, pal)})
    return Piece(glyph, {"in[0]": leg}, ["in[0]"])


def make_identity_piece(state: SceneState, gap_k: int, pal: dict) -> Piece:
    """Identity: the straight wire between the output square and the input circle."""
    endpoints = [f"out[{gap_k}]", f"in[{gap_k + 1}]"]
    sockets = {e: state.socket_centres[e] for e in endpoints}
    ordered = sorted(endpoints, key=lambda e: sockets[e][0])
    left, right = sockets[ordered[0]], sockets[ordered[1]]
    body = make_wire_path(route_points(Route.STRAIGHT, marker_edge(left, right), marker_edge(right, left)), pal["wire"])
    markers = {e: make_marker(e.split("[")[0], True, sockets[e], pal) for e in endpoints}
    glyph = TensorGlyph(f"identity:{gap_k}", body, "identity", markers=markers)
    return Piece(glyph, {}, endpoints)


def make_effect_piece(state: SceneState, pal: dict) -> Piece:
    """Closing effect: left-pointing arrow to the left of the last output, leg running right into the marker."""
    socket = state.socket_centres["out[2]"]
    y = float(socket[1])
    h = LAYOUT["tri_side"] * np.sqrt(3) / 2
    base_x = socket[0] - LAYOUT["marker_r"] - LAYOUT["leg_len"]
    body = make_arrow_body(point(base_x - h / 3, y), LAYOUT["tri_side"], pal["effect"], pointing_right=False)
    attach = point(body.get_right()[0], y)
    leg = make_wire_path(route_points(Route.STRAIGHT, attach, marker_edge(socket, attach)), pal["wire"])
    glyph = TensorGlyph("effect:out", body, "effect", markers={"out[2]": make_marker("out", True, socket, pal)})
    return Piece(glyph, {"out[2]": leg}, ["out[2]"])


# --------------------------------------------------------------------------
# Animation helpers (never call scene.play)
# --------------------------------------------------------------------------


def rigid_path(mob: VMobject, offsets: list[np.ndarray], durations: list[float], rates: list[Callable]) -> UpdateFromAlphaFunc:
    """Move ``mob`` rigidly through waypoints; exact final position independent of frame rate."""
    start = mob.get_center().copy()
    waypoints = [start] + [start + o for o in offsets]
    total = float(sum(durations))
    edges = np.concatenate([[0.0], np.cumsum(durations) / total])

    def update(m: VMobject, alpha: float) -> None:
        i = min(int(np.searchsorted(edges, alpha, side="right")) - 1, len(durations) - 1)
        local = (alpha - edges[i]) / (edges[i + 1] - edges[i])
        s = rates[i](float(np.clip(local, 0.0, 1.0)))
        m.shift(waypoints[i] + (waypoints[i + 1] - waypoints[i]) * s - m.get_center())

    return UpdateFromAlphaFunc(mob, update, run_time=total, rate_func=linear)


def retract(mob: VMobject, toward: str, run_time: float) -> UpdateFromAlphaFunc:
    """Consumed connection writes out towards ``start``, ``end`` or its ``middle``."""
    original = mob.copy()

    def update(m: VMobject, alpha: float) -> None:
        a = float(np.clip(alpha, 0.0, 1.0))
        if toward == "start":
            lo, hi = 0.0, 1.0 - a
        elif toward == "end":
            lo, hi = a, 1.0
        else:
            lo, hi = a / 2, 1.0 - a / 2
        m.pointwise_become_partial(original, lo, max(hi, lo))
        m.set_stroke(opacity=0.0 if a > 0.92 else 1.0)

    return UpdateFromAlphaFunc(mob, update, run_time=run_time, rate_func=smooth)


def shrink_out(mob: VMobject, run_time: float) -> UpdateFromAlphaFunc:
    original = mob.copy()
    centre = mob.get_center().copy()

    def update(m: VMobject, alpha: float) -> None:
        scale = max(1.0 - alpha, 1e-3)
        m.become(original.copy().scale(scale, about_point=centre))
        if alpha > 0.92:
            m.set_stroke(opacity=0.0).set_fill(opacity=0.0)

    return UpdateFromAlphaFunc(mob, update, run_time=run_time, rate_func=smooth)


def interpolate_path(mob: VMobject, start: np.ndarray, end: np.ndarray, run_time: float) -> UpdateFromAlphaFunc:
    def update(m: VMobject, alpha: float) -> None:
        m.set_points(start + (end - start) * alpha)

    return UpdateFromAlphaFunc(mob, update, run_time=run_time, rate_func=smooth)


def emphasis_overlay(path: VMobject, pal: dict) -> VMobject:
    overlay = path.copy()
    overlay.set_stroke(pal["glow"], width=screen_stroke(STYLE["emphasis_px"]), opacity=1.0)
    overlay.set_z_index(Z["emphasis"])
    return overlay


def pulse_ring(centre: np.ndarray, pal: dict, run_time: float) -> tuple[Succession, VMobject]:
    ring = Circle(radius=LAYOUT["marker_r"] * 1.9)
    ring.set_stroke(pal["glow"], width=screen_stroke(STYLE["pulse_px"]))
    ring.set_fill(opacity=0)
    ring.move_to(centre)
    ring.set_z_index(Z["glow"])
    return Succession(Create(ring, run_time=run_time / 2), Uncreate(ring, run_time=run_time / 2)), ring


# --------------------------------------------------------------------------
# Choreography: each plan_* returns a Beat for one panel
# --------------------------------------------------------------------------


def plan_draw_pt(state: SceneState) -> Beat:
    t = TIMING["diagram_draw"]
    bodies = [DrawBorderThenFill(g.body, run_time=t) for g in state.glyphs.values()]
    labels = [Create(g.label, run_time=t) for g in state.glyphs.values() if g.label is not None]
    wires = [Create(w.path, run_time=t) for w in state.wires.values()]
    sockets = [GrowFromCenter(m, run_time=0.4) for m in state.sockets.values()]

    def commit() -> None:
        for g in state.glyphs.values():
            g.status = Status.LIVE
        for w in state.wires.values():
            w.status = Status.LIVE

    return Beat(
        f"{state.panel_id}:draw_pt",
        [Succession(AnimationGroup(*bodies, *labels, *wires), AnimationGroup(*sockets, lag_ratio=0.05))],
        created_ids=tuple(state.glyphs) + tuple(state.wires),
        commit=commit,
    )


def plan_dock_piece(state: SceneState, piece: Piece, pal: dict, name: str) -> Beat:
    """Draw below the docking row, approach a hover point, settle exactly, pulse the sockets."""
    group = piece.group()
    group.shift(DOWN * LAYOUT["spawn_drop"])
    t = TIMING["piece_draw"]
    draw = [Create(piece.glyph.body, run_time=t) if piece.glyph.kind == "identity" else DrawBorderThenFill(piece.glyph.body, run_time=t)]
    draw += [Create(leg, run_time=t) for leg in piece.legs.values()]
    draw += [GrowFromCenter(m, run_time=t) for m in piece.glyph.markers.values()]
    if piece.tail is not None:
        draw.append(Create(piece.tail, run_time=t))
    if piece.glyph.label is not None:
        draw.append(Create(piece.glyph.label, run_time=t))
    hover = LAYOUT["spawn_drop"] - LAYOUT["hover_gap"]
    move = rigid_path(
        group,
        [UP * hover, UP * LAYOUT["spawn_drop"]],
        [TIMING["snap_approach"], TIMING["snap_settle"]],
        [rate_functions.ease_out_cubic, rate_functions.ease_in_out_sine],
    )
    pulses, rings = [], []
    for endpoint in piece.endpoints:
        anim, ring = pulse_ring(state.socket_centres[endpoint], pal, TIMING["snap_pulse"])
        pulses.append(anim)
        rings.append(ring)

    def commit() -> None:
        for endpoint in piece.endpoints:
            offset = piece.glyph.markers[endpoint].get_center() - state.socket_centres[endpoint]
            if np.linalg.norm(offset) > 1e-6:
                raise AssertionError(f"{state.panel_id}:{piece.glyph.id} misaligned at {endpoint}: {offset}")
            state.socket_status[endpoint] = piece.glyph.id
        piece.glyph.status = Status.LIVE
        state.glyphs[piece.glyph.id] = piece.glyph
        state.piece_parts[piece.glyph.id] = [*piece.legs.values(), *([piece.tail] if piece.tail else [])]
        state.transient_ids.add(piece.glyph.id)

    return Beat(
        f"{state.panel_id}:{name}",
        [Succession(AnimationGroup(*draw), move, AnimationGroup(*pulses))],
        moving_ids=(piece.glyph.id,),
        created_ids=(piece.glyph.id,),
        commit=commit,
        discard=rings + [state.sockets[e] for e in piece.endpoints],
    )


def plan_merge(
    state: SceneState,
    name: str,
    source_ids: list[str],
    consumed_wire_ids: list[tuple[str, str]],
    pieces: list[Piece],
    target: TensorGlyph,
    external_port_map: dict[str, tuple[str, PortRef]],
    pal: dict,
    extra_survivors: list[tuple[VMobject, np.ndarray]] | None = None,
    merge_time: float | None = None,
) -> Beat:
    """Local contraction: emphasise and write out internal edges, then one composite becomes the target.

    ``consumed_wire_ids`` pairs a registered wire with its retraction direction;
    piece legs and docked markers are consumed implicitly. Surviving registered
    wires listed in ``external_port_map`` are re-attached to ``target`` ports in
    the same beat via explicit start/end path interpolation.
    """
    merge_time = merge_time or TIMING["merge"]
    consumed_paths = [(state.wires[w].path, toward) for w, toward in consumed_wire_ids]
    for piece in pieces:
        consumed_paths += [(leg, "start") for leg in piece.legs.values()]
    consumed_markers = [m for piece in pieces for m in piece.glyph.markers.values()]
    consumed_markers += [state.sockets[e] for piece in pieces for e in piece.endpoints]
    for wid, _ in consumed_wire_ids:
        b = state.wires[wid].b
        if b.owner_id == "socket" and state.socket_status[b.port_name] == "empty":
            raise AssertionError(f"{wid} consumed while its socket is still open")

    overlays = [emphasis_overlay(path, pal) for path, _ in consumed_paths]
    emphasis = AnimationGroup(*[Create(o, run_time=TIMING["emphasis"]) for o in overlays])
    write_out = AnimationGroup(
        *[retract(path, toward, TIMING["retract"]) for path, toward in consumed_paths],
        *[retract(o, toward, TIMING["retract"]) for o, (_, toward) in zip(overlays, consumed_paths)],
        *[shrink_out(m, TIMING["retract"]) for m in consumed_markers],
    )

    # One coincident target copy per source body: every body visibly slides and
    # morphs onto the target (a bare group-to-single transform pads with invisible
    # copies). The copies are swapped for the single real target after play.
    composite = VGroup(*[state.glyphs[g].body for g in source_ids])
    target_copies = VGroup(*[target.body.copy() for _ in source_ids])
    survivors = []
    for wid, (end, ref) in external_port_map.items():
        wire = state.wires[wid]
        start_pts = wire.path.points.copy()
        a, b = (ref, wire.b) if end == "a" else (wire.a, ref)
        pa = target.port(a.port_name) if a.owner_id == target.id else leg_end(state, a, wire.route_kind)
        pb = target.port(b.port_name) if b.owner_id == target.id else leg_end(state, b, wire.route_kind)
        survivors.append(interpolate_path(wire.path, start_pts, route_points(wire.route_kind, pa, pb), merge_time))
    for path, end_pts in extra_survivors or []:
        survivors.append(interpolate_path(path, path.points.copy(), end_pts, merge_time))
    merge = AnimationGroup(ReplacementTransform(composite, target_copies, run_time=merge_time), *survivors)

    fixed = {
        gid: g.body.get_center().copy()
        for gid, g in state.glyphs.items()
        if g.status == Status.LIVE and gid not in source_ids
    }

    def prepare(scene: Scene) -> None:
        scene.remove(*composite.submobjects)
        scene.add(composite)

    def finish(scene: Scene) -> None:
        scene.remove(target_copies, composite)
        scene.add(target.body)

    def commit() -> None:
        for gid in source_ids:
            state.glyphs[gid].status = Status.CONSUMED
            state.transient_ids.discard(gid)
            state.piece_parts.pop(gid, None)
        for wid, _ in consumed_wire_ids:
            state.wires[wid].status = Status.CONSUMED
        for piece in pieces:
            for e in piece.endpoints:
                state.socket_status[e] = "contracted"
        for wid, (end, ref) in external_port_map.items():
            wire = state.wires[wid]
            if end == "a":
                wire.a = ref
            else:
                wire.b = ref
        target.status = Status.LIVE
        state.glyphs[target.id] = target
        state.accumulator_id = target.id
        for gid, centre in fixed.items():
            if np.linalg.norm(state.glyphs[gid].body.get_center() - centre) > 1e-6:
                raise AssertionError(f"untouched glyph {gid} moved during {name}")

    discard = [p for p, _ in consumed_paths] + overlays + consumed_markers + [composite]
    discard += [m for gid in source_ids for m in state.glyphs[gid].markers.values()]
    return Beat(
        f"{state.panel_id}:{name}",
        [Succession(emphasis, write_out, merge)],
        moving_ids=tuple(source_ids) + tuple(external_port_map),
        consumed_ids=tuple(source_ids) + tuple(w for w, _ in consumed_wire_ids),
        created_ids=(target.id,),
        commit=commit,
        prepare=prepare,
        finish=finish,
        discard=discard,
    )


def make_accumulator(state: SceneState, k: int, pal: dict) -> TensorGlyph:
    """Partial result: a square that still carries the memory frontier and newest output."""
    side = LAYOUT["acc_side"]
    body = make_box(point(core_x(state, k), row_y(state)), side, side, pal["accumulator"], LAYOUT["corner"])
    return TensorGlyph(f"acc:k{k}", body, "accumulator", core_port_specs(side, side))


def plan_absorb_initial_state(state: SceneState, piece: Piece, pal: dict) -> Beat:
    target = make_accumulator(state, 0, pal)
    port_map = {"leg:out[0]": ("a", PortRef(target.id, "out", PortKind.PHYSICAL_OUT))}
    if LAYOUT["n_steps"] > 1:
        port_map["mem:0"] = ("a", PortRef(target.id, "mem_w", PortKind.VIRTUAL))
    return plan_merge(
        state, "absorb_initial_state", [piece.glyph.id, "pt:k0"], [("leg:in[0]", "start")],
        [piece], target, port_map, pal,
    )


def plan_contract_next_core(state: SceneState, identity: Piece, next_k: int, pal: dict) -> Beat:
    """Accumulator and the next core merge. The identity is only the wire between them, so it retracts."""
    identity.legs["channel"] = identity.glyph.body
    # The wire is retracted, not morphed. Mark it consumed so the stationary-glyph check leaves it alone.
    identity.glyph.status = Status.CONSUMED
    target = make_accumulator(state, next_k, pal)
    port_map = {f"leg:out[{next_k}]": ("a", PortRef(target.id, "out", PortKind.PHYSICAL_OUT))}
    if next_k < LAYOUT["n_steps"] - 1:
        port_map[f"mem:{next_k}"] = ("a", PortRef(target.id, "mem_w", PortKind.VIRTUAL))
    consumed = [(f"leg:out[{next_k - 1}]", "start"), (f"leg:in[{next_k}]", "start"), (f"mem:{next_k - 1}", "middle")]
    beat = plan_merge(
        state, f"contract_k{next_k}", [state.accumulator_id, f"pt:k{next_k}"], consumed,
        [identity], target, port_map, pal,
    )
    base_commit = beat.commit

    def commit() -> None:
        base_commit()
        identity.glyph.status = Status.CONSUMED
        state.transient_ids.discard(identity.glyph.id)
        state.piece_parts.pop(identity.glyph.id, None)

    beat.commit = commit
    beat.consumed_ids = beat.consumed_ids + (identity.glyph.id,)
    return beat


def plan_reshape_open_state(state: SceneState, pal: dict) -> Beat:
    """The leftover square becomes a right-facing triangle; its open leg flattens and points left.

    Nothing is contracted. The wire already attached to the square is redrawn in place.
    """
    acc = state.glyphs[state.accumulator_id]
    centre = acc.body.get_center().copy()
    side = LAYOUT["result_tri_side"]
    body = make_arrow_body(centre, side, pal["state"], pointing_right=True)
    base_rel = point(body.get_left()[0] - centre[0], 0.0)
    tip_rel = base_rel + LEFT * LAYOUT["open_stub"]
    target = TensorGlyph("result:state", body, "state", {"base": base_rel, "tip": tip_rel})
    last = f"out[{LAYOUT['n_steps'] - 1}]"
    wire = state.wires[f"leg:{last}"]
    flat = route_points(Route.STRAIGHT, centre + base_rel, centre + tip_rel)
    copy = body.copy()
    morph = ReplacementTransform(acc.body, copy, run_time=TIMING["result_settle"])
    bend = interpolate_path(wire.path, wire.path.points.copy(), flat, TIMING["result_settle"])
    socket = state.sockets[last]

    def finish(scene: Scene) -> None:
        scene.remove(copy)
        scene.add(body)

    def commit() -> None:
        acc.status = Status.CONSUMED
        state.transient_ids.discard(acc.id)
        target.status = Status.LIVE
        state.glyphs[target.id] = target
        state.accumulator_id = target.id
        wire.a = PortRef(target.id, "base", PortKind.OPEN_RESULT)
        wire.b = PortRef(target.id, "tip", PortKind.OPEN_RESULT)
        wire.route_kind = Route.STRAIGHT
        state.socket_status[last] = "open"

    return Beat(
        f"{state.panel_id}:reshape_open_state",
        [AnimationGroup(morph, bend, FadeOut(socket, run_time=TIMING["result_settle"]))],
        moving_ids=(acc.id, wire.id),
        consumed_ids=(acc.id,),
        created_ids=(target.id,),
        commit=commit,
        finish=finish,
        discard=[socket],
    )


def plan_finish_as_scalar(state: SceneState, effect: Piece, pal: dict) -> Beat:
    """The effect consumes the last system leg; the circle has no surviving stubs or markers."""
    body = make_scalar_circle(state.glyphs[state.accumulator_id].body.get_center(), pal["scalar"])
    target = TensorGlyph("result:scalar", body, "scalar")
    last = LAYOUT["n_steps"] - 1
    return plan_merge(
        state, "finish_as_scalar", [state.accumulator_id, effect.glyph.id], [(f"leg:out[{last}]", "start")],
        [effect], target, {}, pal, merge_time=TIMING["result_settle"],
    )


def _result_mobs(state: SceneState) -> list[VMobject]:
    mobs = []
    for glyph in state.glyphs.values():
        if glyph.status == Status.LIVE and glyph.kind in ("state", "scalar"):
            mobs.append(glyph.body)
            mobs.extend(glyph.markers.values())
    mobs.extend(wire.path for wire in state.wires.values() if wire.status == Status.LIVE)
    return mobs


def _bounds(mobs: list[VMobject]) -> tuple[float, float, float, float]:
    return (
        min(mob.get_left()[0] for mob in mobs),
        max(mob.get_right()[0] for mob in mobs),
        min(mob.get_bottom()[1] for mob in mobs),
        max(mob.get_top()[1] for mob in mobs),
    )


def plan_zoom_results(scene: "PTContractionScene") -> Beat:
    """Bring the two results together in the middle of the frame and zoom in by 1.2×."""
    left_mobs = _result_mobs(scene.panels[0])
    right_mobs = _result_mobs(scene.panels[1])
    ll, lr, lb, lt = _bounds(left_mobs)
    rl, rr, rb, rt = _bounds(right_mobs)
    current_gap = rl - lr
    delta = LAYOUT["result_gap"] - current_gap
    shift_sum = -(ll + rr)
    dx_l = (shift_sum - delta) / 2
    dx_r = (shift_sum + delta) / 2
    dy_l = -(lb + lt) / 2
    dy_r = -(rb + rt) / 2
    move = rate_functions.smooth
    anims = [mob.animate(run_time=TIMING["zoom"], rate_func=move).shift(point(dx_l, dy_l)) for mob in left_mobs]
    anims += [mob.animate(run_time=TIMING["zoom"], rate_func=move).shift(point(dx_r, dy_r)) for mob in right_mobs]
    frame = scene.camera.frame
    width = frame.width / 1.2
    anims.append(frame.animate(run_time=TIMING["zoom"], rate_func=move).move_to(point(0.0, 0.0)).set(width=width))
    return Beat(name="zoom_results", animations=anims, discard=[frame])


def plan_centre_result(state: SceneState) -> Beat:
    """Slides the result, with any open output attached to it, to the horizontal centre of its panel."""
    (result,) = [g for g in state.glyphs.values() if g.status == Status.LIVE and g.kind in ("state", "scalar")]
    shift = point(state.origin[0] - result.body.get_center()[0], 0.0)
    attached = [w for w in state.wires.values() if w.status == Status.LIVE and result.id in (w.a.owner_id, w.b.owner_id)]
    mobs = [result.body, *result.markers.values(), *(w.path for w in attached)]
    return Beat(
        f"centre_{result.kind}",
        animations=[m.animate.shift(shift) for m in mobs],
        moving_ids=(result.id, *(w.id for w in attached)),
    )


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------


class PTContractionScene(MovingCameraScene):
    def setup(self) -> None:
        self.theme = active_theme()
        self.pal = PALETTES[self.theme]
        self.poster_time: float | None = None

    # -- lifecycle --------------------------------------------------------

    def play_beat(self, beat: Beat) -> None:
        self.play_parallel_beats(beat)

    def play_parallel_beats(self, *beats: Beat) -> None:
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

    def assert_invariants(self) -> None:
        live = {id(m) for m in self.get_mobject_family_members()}
        for state in self.panels:
            for glyph in state.glyphs.values():
                if glyph.status == Status.LIVE and id(glyph.body) not in live:
                    raise AssertionError(f"{state.panel_id}: live glyph {glyph.id} missing from scene")
                if glyph.status == Status.CONSUMED and id(glyph.body) in live:
                    raise AssertionError(f"{state.panel_id}: consumed glyph {glyph.id} still drawn")
            for wire in state.wires.values():
                if wire.status == Status.LIVE:
                    for ref in (wire.a, wire.b):
                        if ref.owner_id in state.glyphs and state.glyphs[ref.owner_id].status != Status.LIVE:
                            raise AssertionError(f"{state.panel_id}: wire {wire.id} references {ref.owner_id}")
                    if id(wire.path) not in live:
                        raise AssertionError(f"{state.panel_id}: live wire {wire.id} missing from scene")
                elif wire.status == Status.CONSUMED and id(wire.path) in live:
                    raise AssertionError(f"{state.panel_id}: consumed wire {wire.id} still drawn")
        expected = set()
        for state in self.panels:
            for glyph in state.glyphs.values():
                if glyph.status == Status.LIVE:
                    expected |= {id(glyph.body), *(id(m) for m in glyph.markers.values())}
                    if glyph.label is not None:
                        expected |= {id(m) for m in glyph.label.get_family()}
            expected |= {id(w.path) for w in state.wires.values() if w.status == Status.LIVE}
            expected |= {id(m) for parts in state.piece_parts.values() for m in parts}
            expected |= {id(state.sockets[n]) for n, s in state.socket_status.items() if s == "empty"}
        frame_ids = {id(self.camera.frame)}
        ghosts = [
            m for m in self.get_mobject_family_members()
            if m.has_points() and id(m) not in expected and id(m) not in frame_ids
        ]
        if ghosts:
            raise AssertionError(f"{len(ghosts)} unowned mobjects left in the scene: {ghosts[:3]}")

    def assert_identical_panels(self) -> None:
        left, right = self.panels
        rel = lambda s: [s.glyphs[f"pt:k{k}"].body.get_center() - s.origin for k in range(LAYOUT["n_steps"])]
        if not np.allclose(rel(left), rel(right)):
            raise AssertionError("contraction panels do not share core geometry")

    def assert_results(self) -> None:
        left, right = self.panels

        def live(state, kind):
            return [g for g in state.glyphs.values() if g.status == Status.LIVE and g.kind == kind]

        def live_wires(state, prefix=""):
            return [w for w in state.wires.values() if w.status == Status.LIVE and w.id.startswith(prefix)]

        if len(live(left, "state")) != 1 or len(live_wires(left)) != 1 or live_wires(left, "mem"):
            raise AssertionError("left panel must end as one state with one open output and no memory bonds")
        if len(live(right, "scalar")) != 1 or live_wires(right):
            raise AssertionError("right panel must end as one scalar with zero open ports")
        for state in self.panels:
            if [g for g in state.glyphs.values() if g.status == Status.LIVE and g.kind in ("pt_core", "accumulator")]:
                raise AssertionError(f"{state.panel_id}: source cores or accumulators survived")

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
        self.wait(TIMING["blank_hold"])

    def pause(self) -> None:
        self.wait(TIMING["beat_gap"])

    # -- story ------------------------------------------------------------

    def construct(self) -> None:
        pal = self.pal
        origins = LAYOUT["panel_origins"]
        left = make_pt_panel("left", origins["left"], "open_output", pal)
        right = make_pt_panel("right", origins["right"], "effect", pal)
        self.panels = (left, right)
        self.wait(TIMING["blank_hold"])

        self.play_parallel_beats(plan_draw_pt(left), plan_draw_pt(right))
        self.assert_identical_panels()
        self.checkpoint("contraction_two_pts")
        self.pause()

        states = {p.panel_id: make_panel_initial_state(p, pal) for p in self.panels}
        self.play_parallel_beats(*[plan_dock_piece(p, states[p.panel_id], pal, "prepare_and_dock") for p in self.panels])
        self.play_parallel_beats(*[plan_absorb_initial_state(p, states[p.panel_id], pal) for p in self.panels])
        self.pause()

        for next_k in range(1, LAYOUT["n_steps"]):
            ids = {p.panel_id: make_identity_piece(p, next_k - 1, pal) for p in self.panels}
            self.play_parallel_beats(*[plan_dock_piece(p, ids[p.panel_id], pal, f"insert_identity_{next_k - 1}") for p in self.panels])
            self.play_parallel_beats(*[plan_contract_next_core(p, ids[p.panel_id], next_k, pal) for p in self.panels])
            self.checkpoint(f"contraction_after_k{next_k}")
            self.pause()

        effect = make_effect_piece(right, pal)
        self.play_parallel_beats(plan_dock_piece(right, effect, pal, "append_effect"))
        self.play_parallel_beats(plan_reshape_open_state(left, pal), plan_finish_as_scalar(right, effect, pal))
        self.play_parallel_beats(*[plan_centre_result(p) for p in self.panels])
        self.play_beat(plan_zoom_results(self))
        self.assert_results()
        self.checkpoint("contraction_results")

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


def render_assets(scene_cls: type[Scene] = PTContractionScene) -> None:
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
