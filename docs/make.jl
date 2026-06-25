using ProcessTensors
using Documenter

DocMeta.setdocmeta!(ProcessTensors, :DocTestSetup, :(using ProcessTensors); recursive=true)

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
        "Tutorials" => [
            "Liouville Basics" => "tutorials/liouville_basics.md",
            "Unitary Dynamics" => "tutorials/unitary_dynamics.md",
            "Dissipative Dynamics" => "tutorials/dissipative_dynamics.md",
            "Single-Mode Process Tensor" => "tutorials/process_tensor_single_mode.md",
            "Multi-Time Correlations" => "tutorials/multitime_correlations.md",
        ],
        "API Reference" => "api.md",
        "ITensorMPS compatibility" => "itensormps_compatibility.md",
        "Examples" => "examples.md",
        "References" => "references.md",
    ],
)

deploydocs(;
    repo="github.com/Gauthameshwar/ProcessTensors.jl",
    devbranch="main",
)
