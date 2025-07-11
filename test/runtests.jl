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

    @testset "denoise_image" begin
        img = fill(0.5f0, 32, 32)
        noisy = ImageGenerationDiffusionModels.apply_noise(img)
        denoised = ImageGenerationDiffusionModels.denoise_image(noisy)
        @test size(denoised) == (32, 32)
        @test isfile("denoised_img.png")
    end

    @testset "train_brain" begin
        try
            ImageGenerationDiffusionModels.train_brain(1)  # quick smoke test
            @test true
        catch e
            @test false  # fail if any error
        end
    end

    @testset "generate_image_from_noise" begin
        gen = ImageGenerationDiffusionModels.generate_image_from_noise()
        @test size(gen) == (32, 32)
        @test all(0 .<= gen .<= 1)
    end

    @testset "sinusoidal_embedding" begin
        t = [0.0f0, 1.0f0, 2.0f0]
        emb = ImageGenerationDiffusionModels.sinusoidal_embedding(t, 8)
        @test size(emb) == (3, 8)
    end
end