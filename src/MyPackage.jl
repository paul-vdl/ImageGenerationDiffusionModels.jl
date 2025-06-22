module MyPackage

using MAT
using Images
using FileIO
using Flux


# Define the model globally with Float32 types
const model = Chain(
    Dense(32 * 32, 128, relu),  # First layer
    Dense(128, 32 * 32)         # Second layer
)

"""
    function generate_grid()

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
    function apply_noise(img; num_noise_steps = 500, beta_min = 0.0001, beta_max = 0.02)

Applies forward-noise to an image
This function adds Gaussian noise to an image during multiple steps, which corresponds to the forward process in diffusion models.

#Arguments
- 'img' : The input image
- 'num_noise_steps' : number of steps over which noise should be added to the image (500 by default).
- 'beta_min': Minimum beta value (0.0001 by default)
- 'beta_max': Maximum beta value (0.02 by default)

#Returns
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



"""
    function denoise_image(noisy_img)

Denoises a noisy image using the trained neural network 'model'.
Given a single input `noisy_img::Matrix{<:Real}`, this function produces
a denoised version of that input file"""
function denoise_image(noisy_img::AbstractMatrix{<:Real})
  flatten32(mat) = reshape(Float32.(mat), :, 1)
  # Denoise the user’s single image
  x_input = flatten32(noisy_img)
  y_pred  = model(x_input)
  denoised = clamp01.(reshape(y_pred, 32, 32))
  save("denoised_img.png", colorview(Gray, denoised))
  return denoised
end
#TODO do it also in several small steps; do we train a different model per step?

"""
    denoise_image(noisy_img; num_steps=500)


  1. loads the clean 32×32 images,
  2. creates noisy versions of all of them,
  3. trains `model` to map noisy→clean by MSE
"""
function train_brain(num_steps::Int=500)
  # 1) Load the clean images
  data = matread(joinpath(@__DIR__, "..", "SyntheticImages500.mat"))
  raw  = data["syntheticImages"]          # size (32,32,1,500)
  imgs = reshape(raw, 32,32,500)          # now 32×32×500
  clean_images = [imgs[:,:,i] for i in 1:500]  # vector of 32×32 matrices

  # 2) Make noisy versions
  noisy_images = [apply_noise(clean_images[i]) for i in 1:500]

  # 3) Flatten to Float32 column‐vectors
  flatten32(mat) = reshape(Float32.(mat), :, 1)
  clean_vecs = map(flatten32, clean_images)
  noisy_vecs = map(flatten32, noisy_images)

  # 4) Zip into (input,target) pairs
  data_pairs = zip(noisy_vecs, clean_vecs)

  # 5) Set up the optimizer with state
  opt = Flux.setup(ADAM(), model)

  # 6) Define loss
  loss(model, x, y) = Flux.Losses.mse(model(x), y)

  # 7) Train for `num_steps` epochs, always passing `model` to train!
  @info "Training for $num_steps epochs…"
  for epoch in 1:num_steps
    Flux.train!(loss, model, data_pairs, opt)
    if epoch % 10 == 0
      x0,y0 = first(data_pairs)
      @info(" epoch $epoch → training loss = $(loss(model, x0,y0))")
    end # endif
  end #endfor
end #end train_brain

"""
    function generate_image_from_noise()

Generates a new image from random noise and denoises it.
"""
function generate_image_from_noise()
    noisy_img = randn(32, 32)  # Generate random noise
    generated_img = denoise_image(noisy_img)  # "Denoise" the noisy image
    return generated_img  # Return the generated image
end

end  # End of module MyPackage

