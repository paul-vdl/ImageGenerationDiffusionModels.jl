using ImageGenerationDiffusionModels
using Test

@testset "ImageGenerationDiffusionModels.jl" begin
    mat_file = joinpath(@__DIR__, "..", "SyntheticImages500.mat")
    output_file = "grille.png"

    try
        ImageGenerationDiffusionModels.generate_grid()
        @test isfile("grid.png")
    catch e
        @error "Error $e"
        @test false 
    end

    @testset "apply_noise" begin
        img = fill(0.7, 64, 64)
        img_after_noise = ImageGenerationDiffusionModels.apply_noise(img)
        @test !all(img_after_noise .== img)
        @test isfile("noisy_img.png")
    end

end

