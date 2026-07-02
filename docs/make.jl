using ProcessTensors
using Documenter
using Literate

DocMeta.setdocmeta!(ProcessTensors, :DocTestSetup, :(using ProcessTensors); recursive=true)

const DOCS_ROOT = @__DIR__
const PKG_LOGO = normpath(joinpath(DOCS_ROOT, "..", "logo.svg"))
const DOCS_LOGO = joinpath(DOCS_ROOT, "src", "assets", "logo.svg")

# Single source of truth: package-root logo.svg (also used by README).
isfile(PKG_LOGO) || throw(ArgumentError("Package logo not found at $PKG_LOGO"))
mkpath(dirname(DOCS_LOGO))
cp(PKG_LOGO, DOCS_LOGO; force=true)

const LITERATE_DIR = joinpath(DOCS_ROOT, "literate", "tutorials")
const TUTORIAL_OUT = joinpath(DOCS_ROOT, "src", "tutorials")
const LITERATE_EXAMPLE_DIR = joinpath(DOCS_ROOT, "literate", "examples")
const EXAMPLE_OUT = joinpath(DOCS_ROOT, "src", "examples")
const EXAMPLE_ASSETS = joinpath(DOCS_ROOT, "src", "assets", "examples")
const SCRIPT_FIGURES = normpath(joinpath(DOCS_ROOT, "..", "scripts", "figures"))

const TUTORIAL_GROUPS = [
    ("Foundations", [
        ("00_itensor_basics.jl", "itensor_basics", "ITensor Basics"),
        ("01_mps_mpo_basics.jl", "mps_mpo_basics", "MPS and MPO Basics"),
        ("02_liouville_basics.jl", "liouville_basics", "Liouville-Space Basics"),
    ]),
    ("Dynamics", [
        ("03_unitary_dynamics.jl", "unitary_dynamics", "Unitary Dynamics"),
        ("04_dissipative_dynamics.jl", "dissipative_dynamics", "Dissipative Dynamics"),
    ]),
    ("Process tensors", [
        ("05_process_tensor_singlemode.jl", "process_tensor_singlemode", "Single-Mode Process Tensor"),
    ]),
]

const TUTORIALS = vcat((pages for (_, pages) in TUTORIAL_GROUPS)...)

mkpath(TUTORIAL_OUT)

tutorial_stems = Set(stem for (_, stem, _) in TUTORIALS)
for file in readdir(TUTORIAL_OUT)
    if endswith(file, ".md") && file != "README.md"
        stem = replace(file, ".md" => "")
        stem ∉ tutorial_stems && rm(joinpath(TUTORIAL_OUT, file); force=true)
    end
end

for (src, stem, _) in TUTORIALS
    Literate.markdown(
        joinpath(LITERATE_DIR, src),
        TUTORIAL_OUT;
        name=stem,
        documenter=true,
        credit=false,
        execute=true,
    )
end

tutorial_sidebar = [
    group => ["$title" => "tutorials/$stem.md" for (_, stem, title) in pages]
    for (group, pages) in TUTORIAL_GROUPS
]

const LITERATE_EXAMPLES = [
    ("tebd_time_evolution.jl", "tebd_time_evolution", "TEBD time evolution"),
    ("tdvp_time_evolution.jl", "tdvp_time_evolution", "TDVP time evolution"),
    ("spin_bath_process_tensor.jl", "spin_bath_process_tensor", "Spin-bath process tensor"),
    ("multitime_correlations.jl", "multitime_correlations", "Multi-time correlations"),
]

const EXAMPLE_GROUPS = [
    ("Time evolution algorithms", [
        ("TEBD time evolution", "tebd_time_evolution"),
        ("TDVP time evolution", "tdvp_time_evolution"),
    ]),
    ("Dissipative dynamics", [
        ("Dissipative spin", "dissipative_spin"),
        ("Dissipative boson cavity", "dissipative_boson_cavity"),
        ("Boundary-driven chain", "boundary_driven_spin_chain"),
    ]),
    ("Driven systems", [
        ("Driven two-level system", "driven_two_level_system"),
        ("Kicked Ising chain", "kicked_ising_chain"),
    ]),
    ("Process tensors", [
        ("Spin-bath process tensor", "spin_bath_process_tensor"),
    ]),
    ("Instruments and correlations", [
        ("Instrument sequences", "instrument_sequences"),
        ("Multi-time correlations", "multitime_correlations"),
    ]),
]

example_stems = Set(stem for (_, pages) in EXAMPLE_GROUPS for (_, stem) in pages)

example_sidebar = [
    group => ["$title" => "examples/$stem.md" for (title, stem) in pages]
    for (group, pages) in EXAMPLE_GROUPS
]

function stage_example_figures(fig_names)
    mkpath(EXAMPLE_ASSETS)
    for name in fig_names
        src = joinpath(SCRIPT_FIGURES, name)
        isfile(src) || @warn "Example figure not found; run the linked script first." name src
        isfile(src) && cp(src, joinpath(EXAMPLE_ASSETS, name); force=true)
    end
end

mkpath(EXAMPLE_OUT)
literate_example_stems = Set(stem for (_, stem, _) in LITERATE_EXAMPLES)
for file in readdir(EXAMPLE_OUT)
    if endswith(file, ".md")
        stem = replace(file, ".md" => "")
        (stem ∈ literate_example_stems || stem ∉ example_stems) && rm(joinpath(EXAMPLE_OUT, file); force=true)
    end
end

for (src, stem, _) in LITERATE_EXAMPLES
    Literate.markdown(
        joinpath(LITERATE_EXAMPLE_DIR, src),
        EXAMPLE_OUT;
        name=stem,
        documenter=true,
        credit=false,
        execute=true,
    )
end

stage_example_figures([
    "tebd_tfim_unitary_hilbert_dynamics_mx.png",
    "tebd_tfim_unitary_hilbert_rho_error.png",
    "tebd_tfim_unitary_liouville_dynamics_mx.png",
    "tebd_tfim_unitary_liouville_rho_error.png",
    "tdvp_tfim_unitary_hilbert_dynamics_mx.png",
    "tdvp_tfim_unitary_hilbert_energy_drift.png",
    "tdvp_tfim_unitary_hilbert_rho_error.png",
    "tdvp_tfim_unitary_liouville_dynamics_mx.png",
    "tdvp_tfim_unitary_liouville_energy_drift.png",
    "tdvp_tfim_unitary_liouville_rho_error.png",
    "pt_tfim_singlemode.png",
    "pt_tfim_multimode.png",
    "pt_multitime_correlations.png",
])

makedocs(;
    modules=[
        ProcessTensors,
        ProcessTensors.Basis,
        ProcessTensors.Instruments,
        ProcessTensors.Environments,
        ProcessTensors.Spectrals,
    ],
    checkdocs=:none,
    authors="Gauthameshwar <gauthameshwar_s@mymail.sutd.edu.sg> and contributors",
    sitename="ProcessTensors.jl",
    format=Documenter.HTML(;
        canonical="https://Gauthameshwar.github.io/ProcessTensors.jl",
        edit_link="main",
        collapselevel=1,
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Theory" => [
            "Tensor Networks in Physics" => "theory/tensor_networks.md",
            "Quantum States and Liouville Space" => "theory/liouville_space.md",
            "Process Tensors" => "theory/process_tensors.md",
        ],
        "Tutorials" => tutorial_sidebar,
        "Examples" => example_sidebar,
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/Gauthameshwar/ProcessTensors.jl",
    devbranch="main",
)
