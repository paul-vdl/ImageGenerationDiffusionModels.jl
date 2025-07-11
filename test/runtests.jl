using ImageGenerationDiffusionModels
using Test

@testset "ImageGenerationDiffusionModels.jl" begin
    mat_file = joinpath(@__DIR__, "..", "SyntheticImages500.mat")
    output_file = "grille.png"

    try
        generate_grid()
        @test isfile("grid.png")
    catch e
        @error "Error $e"
        @test false 
    end

    @testset "apply_noise" begin
        img = fill(0.7, 64, 64)
        img_after_noise = apply_noise(img)
        @test !all(img_after_noise .== img)
        @test isfile("noisy_img.png")
    end

    @testset "denoise_image" begin
        img = fill(0.5f0, 32, 32)
        noisy = apply_noise(img)
        denoised = denoise_image(noisy)
        @test size(denoised) == (32, 32)
        @test isfile("denoised_img.png")
    end

    @testset "generate_image" begin
        gen = generate_image()
        @test size(gen) == (32, 32)
        @test all(-1 .<= gen .<= 1)
    end
end
