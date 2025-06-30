module MyPackage

using MLDatasets
using Flux
using Flux: @functor, chunk, params
using Parameters: @with_kw
using BSON
using Images
using Logging: with_logger
using ProgressMeter: Progress, next!
using TensorBoardLogger: TBLogger, tb_overwrite
using Random
using Statistics

using NNlib: pad



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

"""
    function generate_image_from_noise()

Generates a new image from random noise and denoises it.
"""
function generate_image_from_noise()
    noisy_img = randn(32, 32)  # Generate random noise
    generated_img = denoise_image(noisy_img)  # "Denoise" the noisy image
    return generated_img  # Return the generated image
end

function sinusoidal_embedding(t::AbstractArray, dim::Int)
    half_dim = div(dim, 2)
    emb = log(10000.0) / (half_dim - 1)
    emb = exp.((-emb) .* (0:half_dim - 1))
    emb = t .* emb'
    emb = hcat(sin.(emb), cos.(emb))
    return emb
end

function pad_or_crop(x, ref)
    _, _, h1, w1 = size(x)
    _, _, h2, w2 = size(ref)
    pad_h = max(0, h2 - h1)
    pad_w = max(0, w2 - w1)
    x = pad(x, (0,0), (0,0), (pad_h÷2, pad_h - pad_h÷2), (pad_w÷2, pad_w - pad_w÷2))
    return x[:, :, 1:h2, 1:w2]
end

function down_block(in_ch, out_ch, time_dim)
    conv1 = Chain(Conv((3,3), in_ch => out_ch, pad=1), BatchNorm(out_ch), relu)
    conv2 = Chain(Conv((3,3), out_ch => out_ch, pad=1), BatchNorm(out_ch), relu)
    downsample = Conv((4,4), out_ch => out_ch, stride=2, pad=1)
    time_mlp = Dense(time_dim, out_ch)

    return (x, t_emb) -> begin
        h = conv1(x)
        t_proj = reshape(relu(time_mlp(t_emb)), :, 1, 1, size(x)[end])
        h = h .+ permutedims(t_proj, (4,1,2,3))
        h = conv2(h)
        return downsample(h), h
    end
end

function up_block(in_ch, out_ch, time_dim)
    upsample = ConvTranspose((4,4), in_ch => in_ch, stride=2, pad=1)
    conv1 = Chain(Conv((3,3), in_ch + div(in_ch,2) => out_ch, pad=1), BatchNorm(out_ch), relu)
    conv2 = Chain(Conv((3,3), out_ch => out_ch, pad=1), BatchNorm(out_ch), relu)
    time_mlp = Dense(time_dim, out_ch)

    return (x, skip, t_emb) -> begin
        x = upsample(x)
        x = pad_or_crop(x, skip)
        x = cat(x, skip; dims=1)
        h = conv1(x)
        t_proj = reshape(relu(time_mlp(t_emb)), :, 1, 1, size(x)[end])
        h = h .+ permutedims(t_proj, (4,1,2,3))
        return conv2(h)
    end
end


function build_unet(in_ch::Int=1, out_ch::Int=1, time_dim::Int=256)
    conv0 = Conv((3,3), in_ch => 128, pad=1)

    down1 = down_block(128, 256, time_dim)
    down2 = down_block(256, 512, time_dim)
    down3 = down_block(512, 1024, time_dim)

    bottleneck = Chain(
        Conv((3,3), 1024 => 1024, pad=1),
        BatchNorm(1024),
        relu,
        Conv((3,3), 1024 => 1024, pad=1),
        BatchNorm(1024),
        relu
    )

    up1 = up_block(1024, 512, time_dim)
    up2 = up_block(512, 256, time_dim)
    up3 = up_block(256, 128, time_dim)

    final = Conv((1,1), 128 => out_ch)

    return (x, t_vec) -> begin
        t_emb = sinusoidal_embedding(t_vec, time_dim)
        x0 = conv0(x)
        x1, skip1 = down1(x0, t_emb)
        x2, skip2 = down2(x1, t_emb)
        x3, skip3 = down3(x2, t_emb)
        x4 = bottleneck(x3)
        x = up1(x4, skip3, t_emb)
        x = up2(x, skip2, t_emb)
        x = up3(x, skip1, t_emb)
        return final(x)
    end
end

"""
Helper function that loads MNIST images and returns loader.
"""
function get_data(batch_size)
    xtrain, ytrain = MLDatasets.MNIST(:train)[:]
    xtrain = reshape(xtrain, 28, 28, 1, :)
    DataLoader((xtrain, ytrain), batchsize=batch_size, shuffle=true)
end


"""
Identical training loop from the example code.
Uses score-matching loss and the same optimizer setup.
"""
function train_brain(; kws...)
    # --- Hyperparameters (copied exactly) ---
    args = (;
        η = 1e-4,        # learning rate
        batch_size = 32, # batch size
        epochs = 50,     # number of epochs
        seed = 1,        # random seed
        cuda = false,    # use CPU (since you don't have GPU)
        verbose_freq = 10,
        save_path = "output"
    )

    Random.seed!(args.seed)
    device = cpu  # Force CPU

    # --- Data Loading (adapted to your dataset) ---
    data = matread(joinpath(@__DIR__, "..", "SyntheticImages500.mat"))
    raw = data["syntheticImages"]
    xtrain = Float32.(reshape(raw, 32, 32, 1, 500))  # Reshape to WHCN format
    batches = [xtrain[:,:,:,i:min(i+args.batch_size-1, end)] 
           for i in 1:args.batch_size:size(xtrain,4)]

    # --- Model Setup (use your U-Net) ---
    model = build_unet() |> device
    opt = ADAM(args.η)
    ps = Flux.params(model)

    # --- Training Loop (identical to example) ---
    @info "Start Training, total $(args.epochs) epochs"
    for epoch in 1:args.epochs
        for x in loader
            x = device(x)
            loss, grad = Flux.withgradient(ps) do
                # Score-matching loss from example
                batch_size = size(x)[end]
                random_t = rand(Float32, batch_size) .* (1.0f0 - 1.0f-5) .+ 1.0f-5
                z = randn(Float32, size(x))
                std = reshape(marginal_prob_std(random_t), 1, 1, 1, :)
                perturbed_x = x + z .* std
                score = model(perturbed_x, random_t)
                mean(sum((score .* std .+ z).^2; dims=1:3))
            end
            Flux.update!(opt, ps, grad)
        end
        @info "Epoch $epoch" loss=loss
    end

    # Save model (optional)
    !ispath(args.save_path) && mkpath(args.save_path)
    model_path = joinpath(args.save_path, "model.bson")
    BSON.@save model_path model
    @info "Model saved: $(model_path)"
end

# Add marginal_prob_std (required for training)
function marginal_prob_std(t, sigma=25.0f0)
    sqrt.((sigma.^(2t) .- 1.0f0) ./ (2.0f0 * log(sigma)))
end






end  # End of module MyPackage

MyPackage.train_brain()
