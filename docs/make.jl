using ProcessTensors
using Documenter

DocMeta.setdocmeta!(ProcessTensors, :DocTestSetup, :(using ProcessTensors); recursive=true)

makedocs(;
    modules=[ProcessTensors],
    authors="Gauthameshwar <gauthameshwar_s@mymail.sutd.edu.sg> and contributors",
    sitename="ProcessTensors.jl",
    format=Documenter.HTML(;
        canonical="https://Gauthameshwar.github.io/ProcessTensors.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Gauthameshwar/ProcessTensors.jl",
    devbranch="main",
)
