module MyPackage

using MAT
using Images
using FileIO

function generate_grid()
    data = matread(joinpath(@__DIR__, "..", "SyntheticImages500.mat"))
    raw = data["syntheticImages"]         # 32x32x1x500
    images = reshape(raw, 32, 32, 500)     # 32x32x500

    first64 = images[:, :, 1:64]
    img_size = size(first64, 1), size(first64, 2)
    canvas = zeros(Float32, 8 * img_size[1], 8 * img_size[2])

    for i in 0:7, j in 0:7
        idx = i * 8 + j + 1
        canvas[i*img_size[1]+1:(i+1)*img_size[1],
               j*img_size[2]+1:(j+1)*img_size[2]] .= first64[:, :, idx]
    end

    canvas2 = clamp01.(canvas)
    save("grille.png", colorview(Gray, canvas2))
    return canvas
end

function apply_noise(img; num_noise_steps = 500, beta_min = 0.0001, beta_max = 0.02)
    variance_schedule = beta_min : 1 / num_noise_steps : beta_max #defines how the model adds noise to the image
    epsilon = randn(size(img)) #gaussian noise

    for beta in variance_schedule #add noise gradually
    img = sqrt(1-beta) .* img + sqrt(beta) .* epsilon
    end
    
    canvas = clamp01.(img) #save the image in .png
    save("noisy_grid.png", colorview(Gray, canvas)) 
end

end