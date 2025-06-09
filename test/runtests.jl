using MyPackage
using Test

@testset "MyPackage.jl" begin
    mat_file = joinpath(@__DIR__, "..", "SyntheticImages500.mat")
    output_file = "grille.png"

    try
        MyPackage.generate_grid(mat_file, output_file)
        @test isfile(output_file)  # Vérifie que l'image a bien été générée
    catch e
        @error "Erreur lors de l'exécution : $e"
        @test false 
    end

    @testset "apply_noise" begin
        img = fill(0.7, 64, 64)
        img_after_noise = MyPackage.apply_noise(img)
        @test !all(img_after_noise .== img)
        @test isfile(noisy_img.png)
    end

end

