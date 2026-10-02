# Copyright © 2026 Gauthameshwar and ProcessTensors.jl contributors
# SPDX-License-Identifier: MIT
#
# File: docs/make.jl
# Contributor: Gauthameshwar S., Cursor AI and PkgTemplates.jl.
#
# Builds the ProcessTensors.jl documentation site using Documenter.jl.
#
# Run with:
#   julia --project=docs docs/make.jl

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
    ("Tensor-network foundations", [
        ("00_itensor_basics.jl", "itensor_basics", "ITensor Basics"),
        ("01_mps_mpo_basics.jl", "mps_mpo_basics", "MPS and MPO Basics"),
        ("02_liouville_basics.jl", "liouville_basics", "Liouville-Space Basics"),
    ]),
    ("Process tensors", [
        ("03_process_tensor_singlemode.jl", "process_tensor_singlemode", "Construct a process tensor"),
        ("04_process_tensor_instruments.jl", "process_tensor_instruments", "Process tensor instruments"),
    ]),
    ("Additional dynamics tools", [
        ("05_unitary_dynamics.jl", "unitary_dynamics", "Unitary Dynamics"),
        ("06_dissipative_dynamics.jl", "dissipative_dynamics", "Dissipative Dynamics"),
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
    ("spin_bath_process_tensor.jl", "spin_bath_process_tensor", "Spin-bath process tensor"),
    ("central_spin_ace.jl", "central_spin_ace", "Central-spin dynamics using ACE"),
    ("thermal_spinboson_ace.jl", "thermal_spinboson_ace", "Thermal spin-boson dynamics using ACE"),
    ("noisy_quantum_circuit_tester.jl", "noisy_quantum_circuit_tester", "Noisy quantum circuit and testers"),
    ("ramsey_povm.jl", "ramsey_povm", "Ramsey readouts as a probe of bath memory"),
    ("multitime_correlations.jl", "multitime_correlations", "Multi-time correlations"),
    ("dissipative_spin.jl", "dissipative_spin", "Dissipative spin chain"),
    ("driven_dissipative_bose_hubbard.jl", "driven_dissipative_bose_hubbard", "Driven-dissipative Bose–Hubbard"),
    ("laser_driven_tdvp.jl", "laser_driven_tdvp", "Laser-driven TDVP dynamics"),
]

const EXAMPLE_GROUPS = [
    ("Process tensors", [
        ("Spin-bath process tensor", "spin_bath_process_tensor"),
        ("Central-spin dynamics using ACE", "central_spin_ace"),
        ("Thermal spin-boson dynamics using ACE", "thermal_spinboson_ace"),
    ]),
    ("Instruments and correlations", [
        ("Testers and noisy quantum qubits", "noisy_quantum_circuit_tester"),
        ("Ramsey POVM measurements", "ramsey_povm"),
        ("Multi-time correlations", "multitime_correlations"),
    ]),
    ("Additional time evolution", [
        ("Dissipative spin chain", "dissipative_spin"),
        ("Driven-dissipative Bose–Hubbard", "driven_dissipative_bose_hubbard"),
        ("Laser-driven TDVP dynamics", "laser_driven_tdvp"),
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
    "laser_driven_tdvp.png",
    "tebd_tfim_dissipative_dynamics_nup.png",
    "tebd_tfim_dissipative_dynamics_mx.png",
    "driven_dissipative_bose_hubbard.png",
    "pt_tfim_singlemode.png",
    "pt_tfim_multimode.png",
    "central_spin_ace.png",
    "thermal_spinboson_ace.png",
    "noisy_quantum_circuit_tester.png",
    "noisy_quantum_circuit_tester_protocol.png",
    "ramsey_povm_protocol.png",
    "ramsey_povm_records.png",
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
        assets=String["assets/admonitions.css"],
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
        "Advanced Usage" => "advanced_usage.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/Gauthameshwar/ProcessTensors.jl",
    devbranch="main",
    push_preview=true,
)
