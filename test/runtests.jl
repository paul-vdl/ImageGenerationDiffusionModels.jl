using Test
using ImageGenerationDiffusionModels
using Statistics: mean
using Flux

@testset "ImageGenerationDiffusionModels.jl" begin

    @testset "generate_grid" begin
        try
            canvas = ImageGenerationDiffusionModels.generate_grid()
            @test isfile("grid.png")
            @test size(canvas) == (256, 256)
        catch e
            @test false "generate_grid failed: $e"
        end
    end

    @testset "apply_noise" begin
        img = fill(0.7f0, 64, 64)
        noisy = ImageGenerationDiffusionModels.apply_noise(img)
        @test size(noisy) == size(img)
        @test !all(noisy .== img)
        @test all(0 .<= noisy .<= 1)
    end

    @testset "denoise_image basic" begin
        img = fill(0.5f0, 32, 32)
        noisy = ImageGenerationDiffusionModels.apply_noise(img)
        denoised = ImageGenerationDiffusionModels.denoise_image(noisy)
        @test size(denoised) == (32, 32)
        @test all(0 .<= denoised .<= 1)
    end

    @testset "generate_image_from_noise" begin
        gen = ImageGenerationDiffusionModels.generate_image_from_noise()
        @test size(gen) == (32, 32)
        @test all(0 .<= gen .<= 1)
    end

    @testset "sinusoidal_embedding" begin
        t = Float32[0, 1, 2]
        emb = ImageGenerationDiffusionModels.sinusoidal_embedding(t, 16)
        @test size(emb) == (3, 16)
    end

    @testset "SimpleUNet output shape" begin
        model = ImageGenerationDiffusionModels.SimpleUNet(1)
        Flux.testmode!(model)  # <- important
        x = rand(Float32, 32, 32, 1, 2)  # batch of 2 images
        t_emb = reshape(ImageGenerationDiffusionModels.timestep_embedding(1), 1,1,:,1)
        t_emb = repeat(t_emb, 1, 1, 1, 2)
        y = model((x, t_emb))
        @test size(y) == (32, 32, 1, 2)
    end

    @testset "pad_or_crop shape match" begin
        x = randn(Float32, 1, 1, 3, 1)
        ref = randn(Float32, 1, 1, 5, 1)
        out = ImageGenerationDiffusionModels.pad_or_crop(x, ref)
        @test size(out) == size(ref)
    end

    @testset "build_unet forward pass" begin
        model = ImageGenerationDiffusionModels.build_unet()
        Flux.testmode!(model)  # <- important
        x = rand(Float32, 32, 32, 1, 1)
        t = [10.0f0]
        y = model(x, t)
        @test size(y) == size(x)
    end

    @testset "train_brain runs" begin
        try
            ImageGenerationDiffusionModels.train_brain(1)
            @test true
        catch e
            @test false "train_brain crashed: $e"
        end
    end
end


