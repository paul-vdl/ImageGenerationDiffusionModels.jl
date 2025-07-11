module ImageGenerationDiffusionModels

using MAT: matread
using Images: clamp01
using Flux: trainable, Chain, Conv, BatchNorm, MaxPool, ADAM, setup, withgradient, update!, ADAM, setup, withgradient, update!
using NNlib; relu
using Statistics: mean

include("train_brain.jl")
include("generate_images.jl")

"""
    generate_grid()

Loads the digits data and generates grid
"""
function generate_grid()
    data = matread(joinpath(@__DIR__, "..", "SyntheticImages500.mat"))
    raw = data["syntheticImages"]        
    images = reshape(raw, 32, 32, 500)     

    first64 = images[:, :, 1:64]
    img_size = size(first64, 1), size(first64, 2)
    canvas = zeros(Float32, 8 * img_size[1], 8 * img_size[2])

    for i in 0:7, j in 0:7
        idx = i * 8 + j + 1
        canvas[i*img_size[1]+1:(i+1)*img_size[1],
               j*img_size[2]+1:(j+1)*img_size[2]] .= first64[:, :, idx]
    end

    canvas2 = clamp01.(canvas)
    save("grid.png", colorview(Gray, canvas2))
    return canvas
end

"""
    apply_noise(img; num_noise_steps = 500, beta_min = 0.0001, beta_max = 0.02)

Applies forward-noise to an image
This function adds Gaussian noise to an image during multiple steps, which corresponds to the forward process in diffusion models.

# Arguments
- `img` : The input image
- `num_noise_steps`: number of steps over which noise should be added to the image (500 by default).
- `beta_min`: Minimum beta value (0.0001 by default)
- `beta_max`: Maximum beta value (0.02 by default)

# Returns
- An image with noise
"""
function apply_noise(img; num_noise_steps = 500, beta_min = 0.0001, beta_max = 0.02)
    
    variance_schedule = beta_min : (beta_max - beta_min) / num_noise_steps : beta_max
    epsilon = randn(size(img))  # Gaussian noise

    for beta in variance_schedule 
        img = sqrt(1-beta) .* img + sqrt(beta) .* epsilon
    end

    canvas = clamp01.(img)
    save("noisy_img.png", colorview(Gray, canvas)) 

    return img  
end

function demo()
    img = generate_grid()
    noisy_img = apply_noise(img)
    #Denoise image
    denoise_image(noisy_img)
    # Generate images
    generate_image(num_images=5)
    @info "Demo completed!"
end

export generate_grid, apply_noise, train, denoise_image, generate_image, demo

end  # End of module ImageGenerationDiffusionModels