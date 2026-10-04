# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: docs/animations/instrument_sequence.py
# Contributor: Gauthameshwar S.
#
# Generates the schematic InstrumentSeq animation (explicit pieces, then a scan
# that fills the one unspecified intervention with an identity).
#
# Run with:
# PT_ANIM_THEME=light docs/animations/.manim_env/bin/python docs/animations/instrument_sequence.py
"""Instrument-sequence animation for the ProcessTensors.jl homepage and README.

Schematic only: no package calls, no numerics. Time runs right to left. The tape
has three propagation intervals; each grey divider carries the PT-facing output
marker ``out[k]`` on its left and input marker ``in[k]`` on its right.

Tested with Manim Community v0.21.0 (Cairo renderer), Python 3.12, ffmpeg 8.0.1.

Render from the repository root::

    PT_ANIM_THEME=light docs/animations/.manim_env/bin/python docs/animations/instrument_sequence.py
    PT_ANIM_THEME=dark  docs/animations/.manim_env/bin/python docs/animations/instrument_sequence.py

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
    UP,
    AnimationGroup,
    CapStyleType,
    Circle,
    Create,
    DrawBorderThenFill,
    FadeOut,
    GrowFromCenter,
    Line,
    Polygon,
    Rectangle,
    RoundedRectangle,
    Scene,
    Succession,
    Text,
    Uncreate,
    UpdateFromAlphaFunc,
    Wait,
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

SCENE_STEM = "instruments"

STYLE = {
    "outline_px": 5.0,
    "wire_px": 4.5,
    "marker_px": 4.5,
    "tape_px": 3.0,
    "pulse_px": 4.0,
    "reference_pixel_width": 960,
}

PALETTES = {
    "light": {
        "outline": "#2b2b2b",
        "wire": "#3a3a3a",
        "muted": "#a3a3a3",
        "tape": "#c4c4c4",
        "input": "#138a8a",
        "output": "#c27c0e",
        "instrument": ("#f7d2c4", "#c0533a"),
        "identity": ("#d5e2f0", "#4a6f96"),
        "state": ("#c8ecea", "#138a8a"),
        "adapter": ("#fdebc8", "#c27c0e"),
        "glow": "#f2a900",
        "debug": "#7a7a7a",
    },
    "dark": {
        "outline": "#e6e6e6",
        "wire": "#cfcfcf",
        "muted": "#7d8a8a",
        "tape": "#566262",
        "input": "#4fd1c5",
        "output": "#f2b544",
        "instrument": ("#a84e36", "#ffb59e"),
        "identity": ("#345a80", "#a9c8ea"),
        "state": ("#1f7a76", "#8ff0e6"),
        "adapter": ("#8a5f12", "#f2b544"),
        "glow": "#ffd166",
        "debug": "#9e9e9e",
    },
}

TIMING = {
    "blank_hold": 0.15,
    "tape_draw": 0.6,
    "marker_draw": 0.45,
    "piece_draw": 0.5,
    "snap_approach": 0.35,
    "snap_settle": 0.12,
    "snap_pulse": 0.3,
    "notice_hold": 0.9,
    "wave_speed": 2.6,
    "highlight": 0.35,
    "identity_draw": 0.45,
    "result_hold": 1.8,
    "loop_fade": 0.4,
}

LAYOUT = {
    "frame_height": 4.5,
    "n_steps": 3,
    "pitch": 2.2,
    "x_right": 2.2,
    "tape_pad": 0.34,
    "tape_half_width": 3.4,
    "separator_width": 0.13,
    "separator_inset": 0.1,
    "marker_dx": 0.36,
    "dock_y": 0.15,
    "marker_r": 0.085,
    "spawn_drop": 0.55,
    "hover_gap": 0.07,
    "corner": 0.09,
    "tri_side": 0.56,
    "leg_len": 0.42,
    "leg_stub": 0.2,
    "wave_sigma": 0.11,
    "wave_half_width": 0.55,
}


def _fit_tape_to_instruments() -> None:
    """Rails sit ``tape_pad`` outside the unitary square, the tallest instrument.

    The square is centred on ``dock_y``. The previous rails left that square
    hanging below the tape, so the added room is on the bottom.
    """
    span = LAYOUT["pitch"] - 2 * LAYOUT["marker_dx"]
    side = span - 2 * LAYOUT["marker_r"] - 2 * LAYOUT["leg_stub"]
    half = side / 2 + LAYOUT["tape_pad"]
    LAYOUT["tape_top"] = LAYOUT["dock_y"] + half
    LAYOUT["tape_bottom"] = LAYOUT["dock_y"] - half


_fit_tape_to_instruments()

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

Z = {"tape": 0, "wave": 0.4, "wire": 1, "body": 2, "marker": 3, "glow": 4, "debug": 5}


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


class EntryKind(Enum):
    PREPARE = "prepare"
    UNSPECIFIED = "unspecified"
    CUSTOM_MAP = "custom_map"
    OPEN_OUTPUT = "open_output"
    DEFAULT_IDENTITY = "default_identity"


@dataclass
class PortRef:
    owner_id: str
    port_name: str
    semantic_kind: PortKind


@dataclass
class TensorGlyph:
    id: str
    body: VMobject
    kind: str
    markers: dict[str, VMobject] = field(default_factory=dict)
    legs: dict[str, VMobject] = field(default_factory=dict)
    port_kinds: dict[str, PortKind] = field(default_factory=dict)
    label: VMobject | None = None
    status: Status = Status.HIDDEN

    def group(self) -> VGroup:
        parts = [self.body, *self.legs.values(), *self.markers.values()]
        if self.label is not None:
            parts.append(self.label)
        return VGroup(*parts)


@dataclass
class ProtocolEntry:
    id: str
    endpoints: list[str]
    kind: EntryKind
    event_x: float = 0.0
    piece_id: str | None = None
    generated: bool = False


@dataclass
class SceneState:
    glyphs: dict[str, TensorGlyph] = field(default_factory=dict)
    sockets: dict[str, VMobject] = field(default_factory=dict)
    socket_centres: dict[str, np.ndarray] = field(default_factory=dict)
    occupancy: dict[str, str | None] = field(default_factory=dict)
    protocol: list[ProtocolEntry] = field(default_factory=list)
    tape_parts: list[VMobject] = field(default_factory=list)
    transient_ids: set[str] = field(default_factory=set)
    wave: VGroup | None = None
    wave_x: float = 0.0


@dataclass
class Beat:
    name: str
    animations: list
    moving_ids: tuple[str, ...] = ()
    consumed_ids: tuple[str, ...] = ()
    created_ids: tuple[str, ...] = ()
    commit: Callable[[], None] = lambda: None
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


def straight_points(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return split_cubic(a, a + (b - a) / 3, a + 2 * (b - a) / 3, b)


def make_wire(points: np.ndarray, color: str) -> VMobject:
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


def marker_boundary(centre: np.ndarray, toward_x: float) -> np.ndarray:
    """Point on the marker where a horizontal leg, coming from ``toward_x``, meets it."""
    sign = 1.0 if toward_x >= centre[0] else -1.0
    return point(centre[0] + sign * LAYOUT["marker_r"], centre[1])


def debug_label(text: str, position: np.ndarray, pal: dict) -> VMobject:
    label = Text(text, font_size=14, color=pal["debug"])
    label.move_to(position)
    label.set_z_index(Z["debug"])
    return label


# --------------------------------------------------------------------------
# Tape and protocol
# --------------------------------------------------------------------------


def separator_x(k: int) -> float:
    return LAYOUT["x_right"] - k * LAYOUT["pitch"]


def socket_position(name: str) -> np.ndarray:
    role, k = name[:-3].rstrip("["), int(name[-2])
    dx = LAYOUT["marker_dx"]
    x = separator_x(k) + (dx if role == "in" else -dx)
    return point(x, LAYOUT["dock_y"])


def make_tape(state: SceneState, pal: dict) -> None:
    half = LAYOUT["tape_half_width"]
    top, bottom = LAYOUT["tape_top"], LAYOUT["tape_bottom"]
    for y in (top, bottom):
        line = Line(point(-half, y), point(half, y))
        line.set_stroke(pal["tape"], width=screen_stroke(STYLE["tape_px"]))
        line.set_cap_style(CapStyleType.ROUND)
        line.set_z_index(Z["tape"])
        state.tape_parts.append(line)
    inset = LAYOUT["separator_inset"]
    for k in range(LAYOUT["n_steps"]):
        bar = RoundedRectangle(
            width=LAYOUT["separator_width"],
            height=top - bottom - 2 * inset,
            corner_radius=LAYOUT["separator_width"] / 2,
        )
        bar.set_fill(pal["muted"], opacity=1.0)
        bar.set_stroke(width=0)
        bar.move_to(point(separator_x(k), (top + bottom) / 2))
        bar.set_z_index(Z["tape"])
        state.tape_parts.append(bar)
        for role in ("out", "in"):
            name = f"{role}[{k}]"
            centre = socket_position(name)
            state.sockets[name] = make_marker(role, False, centre, pal)
            state.socket_centres[name] = centre
            state.occupancy[name] = None
            if DEBUG["show_port_names"]:
                state.tape_parts.append(debug_label(name, centre + point(0, 0.22), pal))


def make_protocol_positions() -> list[ProtocolEntry]:
    """Protocol positions, earliest (rightmost) to latest (leftmost)."""
    protocol = [
        ProtocolEntry("initial", ["in[0]"], EntryKind.PREPARE),
        ProtocolEntry("gap_0", ["out[0]", "in[1]"], EntryKind.UNSPECIFIED),
        ProtocolEntry("gap_1", ["out[1]", "in[2]"], EntryKind.CUSTOM_MAP),
        ProtocolEntry("final", ["out[2]"], EntryKind.OPEN_OUTPUT),
    ]
    for entry in protocol:
        entry.event_x = float(np.mean([socket_position(e)[0] for e in entry.endpoints]))
    return protocol


# --------------------------------------------------------------------------
# Piece factories (built at their docked position; spawn offset applied later)
# --------------------------------------------------------------------------


def make_initial_triangle(piece_id: str, endpoint: str, pal: dict) -> TensorGlyph:
    """Preparation: right-pointing arrow to the right of its input, leg running left into the marker."""
    socket = socket_position(endpoint)
    y = LAYOUT["dock_y"]
    h = LAYOUT["tri_side"] * np.sqrt(3) / 2
    base_x = socket[0] + LAYOUT["marker_r"] + LAYOUT["leg_len"]
    body = make_arrow_body(point(base_x + h / 3, y), LAYOUT["tri_side"], pal["state"], pointing_right=True)
    attach = point(body.get_left()[0], y)
    leg = make_wire(straight_points(attach, marker_boundary(socket, attach[0])), pal["wire"])
    marker = make_marker("in", True, socket, pal)
    return TensorGlyph(
        piece_id, body, "state",
        markers={endpoint: marker}, legs={endpoint: leg},
        port_kinds={endpoint: PortKind.PHYSICAL_IN},
    )


def make_unitary_square(piece_id: str, endpoints: list[str], pal: dict) -> TensorGlyph:
    """Two-leg map: a square on the marker line, with a short straight leg to each marker."""
    sockets = {e: socket_position(e) for e in endpoints}
    y = LAYOUT["dock_y"]
    xs = sorted(s[0] for s in sockets.values())
    inner = (xs[1] - xs[0]) - 2 * LAYOUT["marker_r"]
    side = inner - 2 * LAYOUT["leg_stub"]
    cx = (xs[0] + xs[1]) / 2
    body = make_box(point(cx, y), side, side, pal["instrument"], LAYOUT["corner"])
    legs, markers = {}, {}
    for endpoint, socket in sockets.items():
        direction = np.sign(socket[0] - cx)
        attach = point(cx + direction * side / 2, y)
        legs[endpoint] = make_wire(straight_points(attach, marker_boundary(socket, attach[0])), pal["wire"])
        markers[endpoint] = make_marker(endpoint[:-3].rstrip("["), True, socket, pal)
    return TensorGlyph(
        piece_id, body, "custom", markers=markers, legs=legs,
        port_kinds={endpoints[0]: PortKind.PHYSICAL_OUT, endpoints[1]: PortKind.PHYSICAL_IN},
    )


def make_identity_line(piece_id: str, endpoints: list[str], pal: dict) -> TensorGlyph:
    """Identity: the straight wire between the output square and the input circle, and nothing else."""
    sockets = {e: socket_position(e) for e in endpoints}
    ordered = sorted(endpoints, key=lambda e: sockets[e][0])
    left, right = sockets[ordered[0]], sockets[ordered[1]]
    start = marker_boundary(left, right[0])
    end = marker_boundary(right, left[0])
    body = make_wire(straight_points(start, end), pal["wire"])
    markers = {e: make_marker(e[:-3].rstrip("["), True, sockets[e], pal) for e in endpoints}
    return TensorGlyph(
        piece_id, body, "identity", markers=markers,
        port_kinds={endpoints[0]: PortKind.PHYSICAL_OUT, endpoints[1]: PortKind.PHYSICAL_IN},
    )


def make_final_arrow(piece_id: str, endpoint: str, pal: dict) -> TensorGlyph:
    """Open output: left-pointing arrow to the left of its marker, leg running right into the marker."""
    socket = socket_position(endpoint)
    y = LAYOUT["dock_y"]
    h = LAYOUT["tri_side"] * np.sqrt(3) / 2
    base_x = socket[0] - LAYOUT["marker_r"] - LAYOUT["leg_len"]
    body = make_arrow_body(point(base_x - h / 3, y), LAYOUT["tri_side"], pal["adapter"], pointing_right=False)
    attach = point(body.get_right()[0], y)
    leg = make_wire(straight_points(attach, marker_boundary(socket, attach[0])), pal["wire"])
    marker = make_marker("out", True, socket, pal)
    return TensorGlyph(
        piece_id, body, "open_output",
        markers={endpoint: marker}, legs={endpoint: leg},
        port_kinds={endpoint: PortKind.PHYSICAL_OUT},
    )


def build_piece(entry: ProtocolEntry, pal: dict) -> TensorGlyph:
    piece_id = f"piece:{entry.id}"
    if entry.kind == EntryKind.PREPARE:
        glyph = make_initial_triangle(piece_id, entry.endpoints[0], pal)
    elif entry.kind == EntryKind.CUSTOM_MAP:
        glyph = make_unitary_square(piece_id, entry.endpoints, pal)
    elif entry.kind == EntryKind.OPEN_OUTPUT:
        glyph = make_final_arrow(piece_id, entry.endpoints[0], pal)
    else:
        glyph = make_identity_line(piece_id, entry.endpoints, pal)
    if DEBUG["show_ids"]:
        glyph.label = debug_label(piece_id, glyph.body.get_bottom() + point(0, -0.18), pal)
    return glyph


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
        target = waypoints[i] + (waypoints[i + 1] - waypoints[i]) * s
        m.shift(target - m.get_center())

    return UpdateFromAlphaFunc(mob, update, run_time=total, rate_func=linear)


def pulse_ring(centre: np.ndarray, pal: dict, run_time: float, scale: float = 1.9) -> tuple[Succession, VMobject]:
    ring = Circle(radius=LAYOUT["marker_r"] * scale)
    ring.set_stroke(pal["glow"], width=screen_stroke(STYLE["pulse_px"]))
    ring.set_fill(opacity=0)
    ring.move_to(centre)
    ring.set_z_index(Z["glow"])
    anim = Succession(Create(ring, run_time=run_time / 2), Uncreate(ring, run_time=run_time / 2))
    return anim, ring


def draw_glyph_animations(glyph: TensorGlyph, run_time: float) -> list:
    anims = [Create(glyph.body, run_time=run_time) if glyph.kind == "identity" else DrawBorderThenFill(glyph.body, run_time=run_time)]
    anims += [Create(leg, run_time=run_time) for leg in glyph.legs.values()]
    anims += [GrowFromCenter(m, run_time=run_time) for m in glyph.markers.values()]
    if glyph.label is not None:
        anims.append(Create(glyph.label, run_time=run_time))
    return anims


# --------------------------------------------------------------------------
# Choreography: each plan_* returns a Beat
# --------------------------------------------------------------------------


def plan_draw_tape(state: SceneState) -> Beat:
    sockets = list(state.sockets.values())
    anims = [
        AnimationGroup(*[Create(p, run_time=TIMING["tape_draw"]) for p in state.tape_parts]),
        AnimationGroup(*[GrowFromCenter(s, run_time=TIMING["marker_draw"]) for s in sockets], lag_ratio=0.08),
    ]
    return Beat("draw_tape", [Succession(*anims)], created_ids=tuple(state.sockets))


def plan_draw_piece_below(state: SceneState, glyph: TensorGlyph, run_time: float) -> Beat:
    glyph.group().shift(DOWN * LAYOUT["spawn_drop"])

    def commit() -> None:
        glyph.status = Status.LIVE
        state.glyphs[glyph.id] = glyph

    return Beat(f"draw:{glyph.id}", draw_glyph_animations(glyph, run_time), created_ids=(glyph.id,), commit=commit)


def plan_snap(state: SceneState, glyph: TensorGlyph, endpoints: list[str], pal: dict) -> Beat:
    """Approach a hover point just below the sockets, settle, pulse the receiving sockets."""
    hover = LAYOUT["spawn_drop"] - LAYOUT["hover_gap"]
    move = rigid_path(
        glyph.group(),
        [UP * hover, UP * LAYOUT["spawn_drop"]],
        [TIMING["snap_approach"], TIMING["snap_settle"]],
        [rate_functions.ease_out_cubic, rate_functions.ease_in_out_sine],
    )
    pulses, rings = [], []
    for endpoint in endpoints:
        anim, ring = pulse_ring(state.socket_centres[endpoint], pal, TIMING["snap_pulse"])
        pulses.append(anim)
        rings.append(ring)

    def commit() -> None:
        for endpoint in endpoints:
            centre = glyph.markers[endpoint].get_center()
            if np.linalg.norm(centre - state.socket_centres[endpoint]) > 1e-6:
                raise AssertionError(f"{glyph.id} marker {endpoint} misaligned by {centre - state.socket_centres[endpoint]}")
            state.occupancy[endpoint] = glyph.id

    covered = [state.sockets[e] for e in endpoints]
    return Beat(
        f"snap:{glyph.id}",
        [Succession(move, AnimationGroup(*pulses))],
        moving_ids=(glyph.id,),
        commit=commit,
        discard=rings + covered,
    )


def plan_insert_piece(state: SceneState, entry: ProtocolEntry, glyph: TensorGlyph, pal: dict) -> Beat:
    beat = plan_snap(state, glyph, entry.endpoints, pal)
    snap_commit = beat.commit

    def commit() -> None:
        snap_commit()
        entry.piece_id = glyph.id

    beat.commit = commit
    return beat


def make_gaussian_wave(pal: dict) -> VGroup:
    """Vertical light band. Each column's opacity is a sharp Gaussian of its offset from the peak."""
    sigma = LAYOUT["wave_sigma"]
    half = LAYOUT["wave_half_width"]
    n = 72
    xs = np.linspace(-half, half, n)
    dx = float(xs[1] - xs[0])
    inset = LAYOUT["separator_inset"]
    top = LAYOUT["tape_top"] - inset
    bottom = LAYOUT["tape_bottom"] + inset
    columns = VGroup()
    for x in xs:
        amp = float(np.exp(-0.5 * (float(x) / sigma) ** 2))
        if amp < 0.02:
            continue
        slab = Rectangle(width=dx * 1.08, height=top - bottom, stroke_width=0)
        slab.set_fill(pal["glow"], opacity=amp)
        slab.move_to(point(float(x), 0.5 * (top + bottom)))
        columns.add(slab)
    columns.set_z_index(Z["wave"])
    return columns


def _hold_until_played(animation) -> None:
    """Keep this animation's mobjects out of the scene until the animation itself begins.

    A non-introducer played beside the scan is added immediately, fully drawn.
    """
    animation.introducer = True


def plan_continuous_scan(state: SceneState, pal: dict) -> Beat:
    """Sweep a Gaussian light from right to left without stopping.

    When the peak reaches the first empty output, highlight the open markers,
    draw the identity below them, then snap it onto those timestamps.
    """
    wave = make_gaussian_wave(pal)
    x0 = LAYOUT["tape_half_width"] + LAYOUT["wave_half_width"]
    x1 = -x0
    wave.move_to(point(x0, wave.get_center()[1]))
    state.wave, state.wave_x = wave, x0

    gap = next(entry for entry in state.protocol if entry.kind == EntryKind.UNSPECIFIED)
    trigger = max(gap.endpoints, key=lambda name: state.socket_centres[name][0])
    trigger_x = float(state.socket_centres[trigger][0])
    run_time = (x0 - x1) / TIMING["wave_speed"]
    hit_time = (x0 - trigger_x) / TIMING["wave_speed"]

    glyph = make_identity_line(f"piece:{gap.id}:identity", gap.endpoints, pal)
    identity_group = glyph.group()
    identity_group.shift(DOWN * LAYOUT["spawn_drop"])
    move = rigid_path(wave, [point(x1 - x0, 0)], [run_time], [linear])
    hover = LAYOUT["spawn_drop"] - LAYOUT["hover_gap"]
    snap = rigid_path(
        identity_group,
        [UP * hover, UP * LAYOUT["spawn_drop"]],
        [TIMING["snap_approach"], TIMING["snap_settle"]],
        [rate_functions.ease_out_cubic, rate_functions.ease_in_out_sine],
    )
    _hold_until_played(snap)
    pulses, pulse_rings = [], []
    highlights = []
    for endpoint in gap.endpoints:
        ring = Circle(radius=LAYOUT["marker_r"] * 2.1)
        ring.set_stroke(pal["glow"], width=screen_stroke(STYLE["pulse_px"]))
        ring.set_fill(opacity=0)
        ring.move_to(state.socket_centres[endpoint])
        ring.set_z_index(Z["glow"])
        highlights.append(ring)
        anim, pulse = pulse_ring(state.socket_centres[endpoint], pal, TIMING["snap_pulse"])
        pulses.append(anim)
        pulse_rings.append(pulse)
    covered = [state.sockets[e] for e in gap.endpoints]
    highlight = AnimationGroup(*[Create(ring, run_time=TIMING["highlight"]) for ring in highlights])
    draw = AnimationGroup(*draw_glyph_animations(glyph, TIMING["identity_draw"]))
    arrive = AnimationGroup(snap, *[Uncreate(ring, run_time=TIMING["snap_approach"]) for ring in highlights])
    finish = AnimationGroup(*pulses, *[FadeOut(socket, run_time=0.12) for socket in covered])
    for step in (highlight, draw, arrive, finish):
        _hold_until_played(step)
    identity = Succession(Wait(hit_time), highlight, draw, arrive, finish)
    _hold_until_played(identity)

    def commit() -> None:
        glyph.status = Status.LIVE
        state.glyphs[glyph.id] = glyph
        gap.piece_id = glyph.id
        gap.kind = EntryKind.DEFAULT_IDENTITY
        gap.generated = True
        for endpoint in gap.endpoints:
            centre = glyph.markers[endpoint].get_center()
            if np.linalg.norm(centre - state.socket_centres[endpoint]) > 1e-6:
                raise AssertionError(f"{glyph.id} marker {endpoint} misaligned")
            state.occupancy[endpoint] = glyph.id
        state.transient_ids.discard("wave")
        state.wave = None
        state.wave_x = x1

    return Beat(
        "scan",
        [move, identity],
        moving_ids=("wave", glyph.id),
        created_ids=(glyph.id,),
        consumed_ids=("wave",),
        commit=commit,
        discard=highlights + pulse_rings + covered + [wave],
    )


# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------


class InstrumentSequenceScene(Scene):
    def setup(self) -> None:
        self.theme = active_theme()
        self.pal = PALETTES[self.theme]
        self.state = SceneState()
        self.poster_time: float | None = None

    # -- lifecycle --------------------------------------------------------

    def play_beat(self, beat: Beat) -> None:
        self.play_parallel_beats(beat)

    def play_parallel_beats(self, *beats: Beat) -> None:
        anims = [a for beat in beats for a in beat.animations]
        if anims:
            self.play(*anims)
        for beat in beats:
            beat.commit()
            if beat.discard:
                self.remove(*beat.discard)
            for mob in beat.discard:
                mob.clear_updaters()
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

    def scene_family_ids(self) -> set[int]:
        return {id(m) for m in self.get_mobject_family_members()}

    def assert_invariants(self) -> None:
        live = self.scene_family_ids()
        for glyph in self.state.glyphs.values():
            if glyph.status == Status.LIVE and id(glyph.body) not in live:
                raise AssertionError(f"live glyph {glyph.id} is not in the scene")
        tape_drawn = bool(self.state.tape_parts) and id(self.state.tape_parts[0]) in live
        for name, owner in self.state.occupancy.items():
            if owner is None:
                if tape_drawn and id(self.state.sockets[name]) not in live:
                    raise AssertionError(f"empty socket {name} is not visible")
            else:
                marker = self.state.glyphs[owner].markers[name]
                if np.linalg.norm(marker.get_center() - self.state.socket_centres[name]) > 1e-6:
                    raise AssertionError(f"socket {name} drifted from {owner}")
                if id(self.state.sockets[name]) in live:
                    raise AssertionError(f"covered hollow socket {name} still drawn")

    def assert_counts(self, occupied: int, empty: int, missing_gaps: int) -> None:
        occ = sum(v is not None for v in self.state.occupancy.values())
        emp = sum(v is None for v in self.state.occupancy.values())
        gaps = sum(e.kind == EntryKind.UNSPECIFIED for e in self.state.protocol)
        if (occ, emp, gaps) != (occupied, empty, missing_gaps):
            raise AssertionError(f"expected occupancy {(occupied, empty, missing_gaps)}, got {(occ, emp, gaps)}")

    def mark_poster(self) -> None:
        self.poster_time = self.renderer.time + EXPORT["poster_offset"]

    def reset_loop(self) -> None:
        mobs = list(self.mobjects)
        if mobs:
            self.play(*[FadeOut(m, scale=0.96) for m in mobs], run_time=TIMING["loop_fade"])
        for m in mobs:
            m.clear_updaters()
        self.clear()
        self.state = SceneState()
        if self.mobjects:
            raise AssertionError("reset_loop left mobjects in the scene")
        self.wait(TIMING["blank_hold"])

    # -- story ------------------------------------------------------------

    def construct(self) -> None:
        state, pal = self.state, self.pal
        make_tape(state, pal)
        state.protocol = make_protocol_positions()
        self.wait(TIMING["blank_hold"])

        self.play_beat(plan_draw_tape(state))
        for entry in state.protocol:
            if entry.kind == EntryKind.UNSPECIFIED:
                continue
            glyph = build_piece(entry, pal)
            self.play_beat(plan_draw_piece_below(state, glyph, TIMING["piece_draw"]))
            self.play_beat(plan_insert_piece(state, entry, glyph, pal))
        self.assert_counts(occupied=4, empty=2, missing_gaps=1)
        self.checkpoint("sequence_before_defaults")
        self.wait(TIMING["notice_hold"])

        self.scan_and_fill_defaults()
        self.assert_counts(occupied=6, empty=0, missing_gaps=0)
        final = next(e for e in state.protocol if e.kind == EntryKind.OPEN_OUTPUT)
        final_glyph = state.glyphs[final.piece_id]
        if final.endpoints[0] not in final_glyph.legs:
            raise AssertionError("open output lost the leg on its right")
        self.checkpoint("sequence_after_defaults")

        self.mark_poster()
        self.wait(TIMING["result_hold"])
        self.reset_loop()

    def scan_and_fill_defaults(self) -> None:
        self.play_beat(plan_continuous_scan(self.state, self.pal))


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


def render_assets(scene_cls: type[Scene] = InstrumentSequenceScene) -> None:
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
