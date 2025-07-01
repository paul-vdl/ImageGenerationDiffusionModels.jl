using MyPackage
using Documenter

DocMeta.setdocmeta!(MyPackage, :DocTestSetup, :(using MyPackage); recursive=true)

makedocs(;
    modules=[MyPackage],
    authors="Paul Vidal <p.vidal@campus.tu-berlin.de>",
    sitename="MyPackage.jl",
    format=Documenter.HTML(;
        canonical="https://paul-vdl.github.io/MyPackage.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "test.md",
    ],
)

deploydocs(;
    repo="github.com/paul-vdl/MyPackage.jl",
    devbranch="master",
)
