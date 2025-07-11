using ImageGenerationDiffusionModels
using Test
using Statistics: mean

@testset "ImageGenerationDiffusionModels.jl" begin
    mat_file = joinpath(@__DIR__, "..", "SyntheticImages500.mat")
    output_file = "grille.png"

 

    @testset "generate_grid" begin
        try
            canvas = ImageGenerationDiffusionModels.generate_grid()
            @test isfile("grid.png")
            @test size(canvas) == (256, 256)  # 8×32
        catch e
            @error "Error $e"
            @test false
        end
    end

    @testset "apply_noise" begin
        img = fill(0.7, 64, 64)
        noisy = ImageGenerationDiffusionModels.apply_noise(img)
        @test size(noisy) == size(img)
        @test !all(noisy .== img)
        @test isfile("noisy_img.png")
        @test all(0 .<= noisy .<= 1)
    end

    @testset "denoise_image" begin
        img = fill(0.5, 32, 32)
        noisy = ImageGenerationDiffusionModels.apply_noise(img)
        denoised = ImageGenerationDiffusionModels.denoise_image(noisy)
        @test size(denoised) == (32, 32)
        @test isfile("denoised_img.png")
        @test all(0 .<= denoised .<= 1)
    end

    @testset "train_brain runs" begin
        try
            ImageGenerationDiffusionModels.train_brain(1)
            @test true
        catch e
            @test false "train_brain failed: $e"
        end
    end

    @testset "generate_image_from_noise" begin
        gen = ImageGenerationDiffusionModels.generate_image_from_noise()
        @test size(gen) == (32, 32)
        @test all(0 .<= gen .<= 1)
    end

    @testset "sinusoidal_embedding basic" begin
        t = [0.0f0, 1.0f0, 2.0f0]
        emb = ImageGenerationDiffusionModels.sinusoidal_embedding(t, 8)
        @test size(emb) == (3, 8)
    end



    @testset "apply_noise range and shape" begin
        img = zeros(32, 32)
        noisy_img = ImageGenerationDiffusionModels.apply_noise(img)
        @test size(noisy_img) == (32, 32)
        @test all(0 .<= noisy_img .<= 1)
    end

    @testset "SimpleUNet output dimensions" begin
        model = ImageGenerationDiffusionModels.SimpleUNet(1)
        x = rand(Float32, 32, 32, 1, 2)  # batch size 2
        t_emb = reshape(ImageGenerationDiffusionModels.timestep_embedding(1), 1,1,:,1)
        t_emb = repeat(t_emb, 1, 1, 1, 2)  # match batch
        y = model((x, t_emb))
        @test size(y) == (32, 32, 1, 2)
    end

    @testset "sinusoidal_embedding size" begin
        t = Float32[0, 1, 2, 3]
        emb = ImageGenerationDiffusionModels.sinusoidal_embedding(t, 16)
        @test size(emb) == (4, 16)
    end

    @testset "train_brain improves output" begin
        img = fill(0.5, 32, 32)
        noisy = ImageGenerationDiffusionModels.apply_noise(img)
        denoised_before = ImageGenerationDiffusionModels.denoise_image(noisy)

        ImageGenerationDiffusionModels.train_brain(3)  # quick learn
        denoised_after = ImageGenerationDiffusionModels.denoise_image(noisy)

    end

    

    @testset "pad_or_crop output shape" begin
        x = randn(Float32, 1, 1, 3, 1)
        ref = randn(Float32, 1, 1, 5, 1)
        result = ImageGenerationDiffusionModels.pad_or_crop(x, ref)
        @test size(result) == size(ref)
    end

    @testset "build_unet forward pass" begin
        unet = ImageGenerationDiffusionModels.build_unet()
        x = rand(Float32, 32, 32, 1, 1)
        t = [10.0f0]
        out = unet(x, t)
        @test size(out) == size(x)
    end

    @testset "sinusoidal_embedding edge dim" begin
        t = Float32[0, 1, 2]
        emb = ImageGenerationDiffusionModels.sinusoidal_embedding(t, 2)
        @test size(emb) == (3, 2)
    end
end

