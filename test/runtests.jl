using Test
using ImageGenerationDiffusionModels

@testset "ImageGenerationDiffusionModels.jl" begin

    @testset "Grid Generation" begin
        try
            canvas = generate_grid()
            @test typeof(canvas) <: AbstractArray
            @test isfile("grid.png")
        catch e
            @error "Grid generation failed: $e"
            @test false
        end
    end

    @testset "Noise Application" begin
        img = fill(0.5f0, 32, 32)
        try
            noisy = apply_noise(img)
            @test typeof(noisy) <: AbstractArray
            @test !all(noisy .== img)
            @test isfile("noisy_img.png")
        catch e
            @error "Noise application failed: $e"
            @test false
        end
    end

    @testset "Denoising" begin
        img = fill(0.5f0, 32, 32)
        noisy = apply_noise(img)
        try
            denoised = denoise_image(noisy)
            @test size(denoised) == (32, 32, 1, 1)
            @test isfile("denoised_image.png")
        catch e
            @error "Denoising failed: $e"
            @test false
        end
    end

    @testset "Model Training (Smoke Test)" begin
        try
            train(joinpath(@__DIR__, "..", "SyntheticImages500.mat"), epochs=1, patience=0)
            @test isfile("trained_model.bson")
            @test isfile("training_loss.png")
        catch e
            @error "Training failed: $e"
            @test false
        end
    end

    @testset "Image Generation" begin
        try
            images = generate_image(num_images=2)
            @test size(images, 1:2) == (32, 32)
            @test size(images, 4) == 2
            @test all(images .>= -1) && all(images .<= 1)
            @test isfile("generated_image_1.png")
        catch e
            @error "Image generation failed: $e"
            @test false
        end
    end

    @testset "Timestep Embedding" begin
        emb = timestep_embedding(10)
        @test length(emb) == D
    end
end
