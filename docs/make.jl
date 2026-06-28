using ProcessTensors
using Documenter
using Literate

DocMeta.setdocmeta!(ProcessTensors, :DocTestSetup, :(using ProcessTensors); recursive=true)

const DOCS_ROOT = @__DIR__
const LITERATE_DIR = joinpath(DOCS_ROOT, "literate", "tutorials")
const TUTORIAL_OUT = joinpath(DOCS_ROOT, "src", "tutorials")

const TUTORIALS = [
    ("00_itensor_basics.jl", "itensor_basics", "ITensor Basics"),
    ("01_mps_mpo_basics.jl", "mps_mpo_basics", "MPS and MPO Basics"),
    ("02_liouville_basics.jl", "liouville_basics", "Liouville-Space Basics"),
    ("03_unitary_dynamics.jl", "unitary_dynamics", "Unitary Dynamics"),
    ("04_dissipative_dynamics.jl", "dissipative_dynamics", "Dissipative Dynamics"),
    ("05_time_dependent_dynamics.jl", "time_dependent_dynamics", "Time-Dependent Dynamics"),
    ("06_process_tensor_singlemode.jl", "process_tensor_singlemode", "Single-Mode Process Tensor"),
    ("07_process_tensor_multimode.jl", "process_tensor_multimode", "Multimode Process Tensor"),
    ("08_multitime_correlations.jl", "multitime_correlations", "Multi-Time Correlations"),
]

mkpath(TUTORIAL_OUT)

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

tutorial_pages = ["$title" => "tutorials/$stem.md" for (_, stem, title) in TUTORIALS]

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
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Theory" => [
            "Tensor Networks in Physics" => "theory/tensor_networks.md",
            "Quantum States and Liouville Space" => "theory/liouville_space.md",
            "Process Tensors" => "theory/process_tensors.md",
        ],
        "Tutorials" => tutorial_pages,
        "API Reference" => "api.md",
        "Examples" => "examples.md",
        "References" => "references.md",
    ],
)

deploydocs(;
    repo="github.com/Gauthameshwar/ProcessTensors.jl",
    devbranch="main",
)
