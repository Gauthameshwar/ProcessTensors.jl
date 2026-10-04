#!/usr/bin/env python3
# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: docs/tikz/generate_dark_svgs.py
# Contributor: Gauthameshwar S.
#
# Generates dark-theme SVG variants from light-theme TikZ SVGs for Documenter docs.
# Uses a curated, soft and vibrant palette for tensor cores, instruments, and operators
# on dark canvas (#1f2424) with crisp off-white labels and outlines (#e6e6e6).

import os
import re
import colorsys
DOC_DARK_BG = "#1f2424"

# Curated dark-theme-friendly palette for ProcessTensors theory figures.
# Preserves semantic color identities with soft, luminous, and vibrant tones.
DARK_PALETTE = {
    # 1. Page knockouts, double-wire inner rails, crossing halos, prime-tick knockouts
    "#ffffff": DOC_DARK_BG,
    "#fff": DOC_DARK_BG,
    "#d1d1d1": DOC_DARK_BG,  # Inkscape desk background

    # 2. Main strokes, wires, labels, text
    "#000000": "#e6e6e6",
    "#000": "#e6e6e6",
    "#333333": "#e6e6e6",

    # 3. Bonds and dividers
    "#6a6a6a": "#9e9e9e",
    "#adadad": "#6e6e6e",

    # 4. Process-tensor cores (W and Q)
    "#dceaf7": "#2b6ea6",  # PTProcess: soft vibrant sapphire / sky core (W)
    "#ace8ae": "#2d854c",  # QAMPOcore: soft vibrant emerald / mint-sage core (Q)

    # 5. Instruments and interventions
    "#f6e8a6": "#997116",  # PTInstrument: soft warm golden amber (E, A)
    "#f7e58b": "#a8801a",  # PTInstIn: warm golden sunflower (instrument input)
    "#fcb06d": "#b55e26",  # PTInstOut: soft vibrant coral / terracotta (instrument output)
    "#ad9986": "#7a6652",  # PTInstMap: refined warm bronze / taupe (instrument map)

    # 6. Generic operators and Liouville superoperators
    "#e8e8e8": "#3e4c52",  # PTTensor: soft elevated slate card (generic operator)
    "#eadeee": "#7b488d",  # PTLiouville: soft vibrant amethyst / purple
    "#e5d6ea": "#7b488d",  # PTLiouville variant

    # 7. Ket and Bra boundary states
    "#b8d8e9": "#2b6ea6",  # PTHilbertKet!18: soft sapphire ket triangle
    "#edbb94": "#a85c2c",  # PTHilbertBra!18: soft terracotta bra triangle
    "#f1c8a8": "#a85c2c",  # PTHilbertBra!18 variant

    # 8. Environment memory loop and bath contours
    "#d1e3d1": "#285236",  # Green environment wash (fill)
    "#004700": "#3eb066",  # Green memory stroke (vibrant mint line)

    # 9. MPS site gradient in mps_vocabulary (3-step blue gradient)
    "#f4f9fc": "#23496d",  # MPS Site A (lightest)
    "#d6e8f4": "#285b88",  # MPS Site B (mid)
    "#b7d4e8": "#2f71a6",  # MPS Site C (deepest)

    # 10. MPO site gradient in mps_vocabulary (3-step warm amber gradient)
    "#fdf8f3": "#5e3d24",  # MPO Site A (lightest)
    "#f6e6d6": "#754b2b",  # MPO Site B (mid)
    "#ebd4be": "#8f5c34",  # MPO Site C (deepest)
}

def hex_to_rgb(h):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(2 * c for c in h)
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

def rgb_to_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(
        int(round(max(0.0, min(1.0, r)) * 255)),
        int(round(max(0.0, min(1.0, g)) * 255)),
        int(round(max(0.0, min(1.0, b)) * 255)),
    )

def transform_color_for_dark(hex_str, dark_bg=DOC_DARK_BG):
    h_lower = hex_str.lower()
    if h_lower in DARK_PALETTE:
        val = DARK_PALETTE[h_lower]
        return dark_bg if val == DOC_DARK_BG else val

    # Graceful fallback for any unlisted color
    r, g, b = hex_to_rgb(h_lower)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    if l > 0.65:
        # Soft vibrant luminous fill
        new_l = 0.35 + (l - 0.65) * 0.15
        new_s = min(1.0, s * 0.85)
    elif l < 0.35:
        new_l = 0.60
        new_s = min(1.0, s * 0.85)
    else:
        new_l = 1.0 - l
        new_s = s

    nr, ng, nb = colorsys.hls_to_rgb(h, new_l, new_s)
    return rgb_to_hex(nr, ng, nb)

def convert_svg_to_dark(src_path, dst_path, dark_bg=DOC_DARK_BG):
    with open(src_path, "r", encoding="utf-8") as f:
        content = f.read()

    def replace_hex(match):
        original_hex = match.group(0)
        return transform_color_for_dark(original_hex, dark_bg)

    # Match #xxxxxx or #xxx hex colors
    new_content = re.sub(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b", replace_hex, content)

    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(new_content)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.normpath(os.path.join(script_dir, "..", "src", "assets", "theory"))

    figures = [
        "pt_anatomy",
        "quantum_channel",
        "channel_to_multitime_process",
        "column_major_vectorisation",
        "mpo_to_liouville_mps",
        "closing_a_leg",
        "process_contractions",
        "tensor_contractions",
        "mps_vocabulary",
    ]

    for name in figures:
        src = os.path.join(assets_dir, f"{name}-light.svg")
        if not os.path.exists(src):
            print(f"Skipping {src} (not found)")
            continue

        dark_dst = os.path.join(assets_dir, f"{name}-dark.svg")
        convert_svg_to_dark(src, dark_dst)
        print(f"Generated {name}-dark.svg")

if __name__ == "__main__":
    main()
