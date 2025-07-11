using ImageGenerationDiffusionModels
using Test
using Statistics: mean

@testset "ImageGenerationDiffusionModels.jl" begin
    mat_file = joinpath(@__DIR__, "..", "SyntheticImages500.mat")
    output_file = "grille.png"

    # ========== Basic Function Tests ========== #

    @testset "generate_grid" begin
        try
            ImageGenerationDiffusionModels.generate_grid()
            @test isfile("grid.png")
        catch e
            @error "Error $e"
            @test false
        end
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
            ImageGenerationDiffusionModels.train_brain(1)  # quick test
            @test true
        catch e
            @test false
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

    # ========== Additional Behavior & Sanity Tests ========== #

    @testset "apply_noise range and shape" begin
        img = zeros(32, 32)
        noisy_img = ImageGenerationDiffusionModels.apply_noise(img)
        @test size(noisy_img) == (32, 32)
        @test all(0 .<= noisy_img .<= 1)
    end

    @testset "generate_image_from_noise validity" begin
        gen = ImageGenerationDiffusionModels.generate_image_from_noise()
        @test size(gen) == (32, 32)
        @test all(0 .<= gen .<= 1)
    end

    @testset "sinusoidal_embedding correctness" begin
        t = Float32[0, 1, 2, 3]
        emb = ImageGenerationDiffusionModels.sinusoidal_embedding(t, 16)
        @test size(emb) == (4, 16)
    end

    @testset "SimpleUNet output dimensions" begin
        model = ImageGenerationDiffusionModels.SimpleUNet(1)
        x = rand(Float32, 32, 32, 1, 2)  # batch size = 2
        t_emb = reshape(ImageGenerationDiffusionModels.timestep_embedding(1), 1,1,:,1)
        t_emb = repeat(t_emb, 1, 1, 1, 2)  # match batch
        y = model((x, t_emb))
        @test size(y) == (32, 32, 1, 2)
    end

    @testset "train_brain improves model" begin
        img = fill(0.5, 32, 32)
        noisy = ImageGenerationDiffusionModels.apply_noise(img)
        denoised_before = ImageGenerationDiffusionModels.denoise_image(noisy)

        ImageGenerationDiffusionModels.train_brain(3)
        denoised_after = ImageGenerationDiffusionModels.denoise_image(noisy)

        mse_before = mean((denoised_before .- img).^2)
        mse_after = mean((denoised_after .- img).^2)

        @test mse_after < mse_before
    end
end
