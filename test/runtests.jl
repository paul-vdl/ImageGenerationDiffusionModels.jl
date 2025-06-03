using MyPackage
using Test

@testset "MyPackage.jl" begin
    mat_file = joinpath(@__DIR__, "..", "SyntheticImages500.mat")
    output_file = "grille.png"

    try
        MyPackage.generate_grid(mat_file, output_file)
        @test isfile(output_file)  # Vérifie que l'image a bien été générée
    catch e
        @test false "Erreur lors de l'exécution : $e"
    end
end
