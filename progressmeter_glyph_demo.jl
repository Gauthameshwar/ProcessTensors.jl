using ProgressMeter

# ─────────────────────────────────────────────
# A quick gallery of BarGlyphs styles to preview.
# Run this script and watch each bar animate.
# Pick your favorite and hardcode it into your package.
# ─────────────────────────────────────────────

styles = [
    ("Classic default",     BarGlyphs("[=> ]")),
    ("Slim line",            BarGlyphs('┣', '━', ' ', '┫', ' ')),
    ("Sub-block smooth",     BarGlyphs('█', '▊', '▌', ' ')),
    ("Dots",                 BarGlyphs('[', '●', '○', '-', ']')),
    ("Shaded gradient",      BarGlyphs('╢', '▓', '▒', '░', '╟')),
    ("ASCII safe",           BarGlyphs('[', '#', '-', ' ', ']')),
]

n = 40  # number of steps per bar, tweak for speed/smoothness

for (name, glyphs) in styles
    println("\n▶ Style: $name")
    p = Progress(n; barglyphs=glyphs, barlen=30, color=:cyan)
    for i in 1:n
        sleep(0.02)  # simulate work — swap for your actual tensor contraction step
        next!(p)
    end
end

println("\nDone! Copy the BarGlyphs(...) line you liked best into your package.")

# ─────────────────────────────────────────────
# Bonus: Braille spinner for indeterminate phases
# (e.g. while contracting an MPO with unknown step count)
# ─────────────────────────────────────────────
function spinner_demo(duration_s=3)
    frames = collect("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
    t0 = time()
    i = 1
    while time() - t0 < duration_s
        print("\r$(frames[i]) working...")
        flush(stdout)
        i = i % length(frames) + 1
        sleep(0.08)
    end
    println("\r✔ done!            ")
end

println("\n▶ Spinner demo:")
spinner_demo()
