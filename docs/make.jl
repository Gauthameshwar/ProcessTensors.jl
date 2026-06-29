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
    ("05_process_tensor_singlemode.jl", "process_tensor_singlemode", "Single-Mode Process Tensor"),
]

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

tutorial_pages = ["$title" => "tutorials/$stem.md" for (_, stem, title) in TUTORIALS]

const EXAMPLES = [
    ("Unitary spin chain", "spin_chain_unitary"),
    ("Bose-Hubbard dynamics", "bose_hubbard_unitary"),
    ("Reduced states and entropy", "reduced_density_entropy"),
    ("Dissipative spin", "dissipative_spin"),
    ("Dissipative boson cavity", "dissipative_boson_cavity"),
    ("Boundary-driven chain", "boundary_driven_spin_chain"),
    ("Driven two-level system", "driven_two_level_system"),
    ("Kicked Ising chain", "kicked_ising_chain"),
    ("Single spin-bath process tensor", "single_spin_bath_process_tensor"),
    ("Stochastic process tensor", "stochastic_process_tensor"),
    ("Multimode process tensor", "multimode_process_tensor"),
    ("Instrument sequences", "instrument_sequences"),
    ("Multi-time correlations", "multitime_correlations"),
    ("Convergence and truncation", "convergence_and_truncation"),
]

example_pages = ["$title" => "examples/$stem.md" for (title, stem) in EXAMPLES]

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
        "Examples" => example_pages,
        "References" => "references.md",
    ],
)

deploydocs(;
    repo="github.com/Gauthameshwar/ProcessTensors.jl",
    devbranch="main",
)
