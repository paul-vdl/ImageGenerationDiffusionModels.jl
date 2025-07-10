using ImageGenerationDiffusionModels
using Documenter

DocMeta.setdocmeta!(ImageGenerationDiffusionModels, :DocTestSetup, :(using ImageGenerationDiffusionModels); recursive=true)

makedocs(;
    modules=[ImageGenerationDiffusionModels],
    authors="Paul Vidal <p.vidal@campus.tu-berlin.de>",
    sitename="ImageGenerationDiffusionModels.jl",
    format=Documenter.HTML(;
        canonical="https://paul-vdl.github.io/ImageGenerationDiffusionModels.jl/dev",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/paul-vdl/ImageGenerationDiffusionModels.jl.git",
    devbranch="main",
    branch="gh-pages"
)
