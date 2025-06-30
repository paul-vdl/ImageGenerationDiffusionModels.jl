module MyPackage

using Flux
using MLDatasets
using BSON: @save, @load
using Images, ImageIO, FileIO
using ProgressMeter: @showprogress
using MAT
using Statistics

f32(x) = Flux.f32(x)

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
    images = Float32.(reshape(raw, 32, 32, 1, 500))   

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
flatten32(mat) = reshape(Float32.(mat), :, 1)
function denoise_image(noisy_img::AbstractMatrix{<:Real})
  
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
flatten32(mat) = reshape(Float32.(mat), :, 1)
function train_brain(num_steps::Int=500)
  # 1) Load the clean images
  data = matread(joinpath(@__DIR__, "..", "SyntheticImages500.mat"))
  raw  = data["syntheticImages"]          # size (32,32,1,500)
  imgs = reshape(raw, 32,32,500)          # now 32×32×500
  clean_images = [imgs[:,:,i] for i in 1:500]  # vector of 32×32 matrices

  # 2) Make noisy versions
  noisy_images = [apply_noise(clean_images[i]) for i in 1:500]

  # 3) Flatten to Float32 column‐vectors
  
  clean_vecs = map(flatten32, clean_images)
  noisy_vecs = map(flatten32, noisy_images)

  # 4) Zip into (input,target) pairs
  data_pairs = zip(noisy_vecs, clean_vecs)

  # 5) Set up the optimizer with state
  opt = ADAM()

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

function sinusoidal_embedding(t::Vector{Float32}, dim::Int)
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
        t_proj = permutedims(reshape(t_proj, (size(t_proj, 2), 1, 1, size(t_proj, 1))), (4,1,2,3))
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

"""
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

function build_unet(in_ch::Int=1, out_ch::Int=1, time_dim::Int=256)
    # Initial convolution (maintains 32x32)
    conv0 = Chain(
        Conv((3,3), in_ch => 64, pad=SamePad()),
        BatchNorm(64),
        x -> relu.(x)
    ) |> f32

    # Downsample path
    down1 = Chain(
        Conv((3,3), 64 => 128, pad=SamePad()),
        BatchNorm(128),
        x -> relu.(x),
        Conv((4,4), 128 => 128, stride=2, pad=1)  # 32x32 -> 16x16
    ) |> f32

    down2 = Chain(
        Conv((3,3), 128 => 256, pad=SamePad()),
        BatchNorm(256),
        x -> relu.(x),
        Conv((4,4), 256 => 256, stride=2, pad=1)  # 16x16 -> 8x8
    ) |> f32

    # Bottleneck (maintains 8x8)
    bottleneck = Chain(
        Conv((3,3), 256 => 512, pad=SamePad()),
        BatchNorm(512),
        x -> relu.(x),
        Conv((3,3), 512 => 512, pad=SamePad()),
        BatchNorm(512),
        x -> relu.(x)
    ) |> f32

    # Upsample path
    up1 = Chain(
        ConvTranspose((4,4), 512 => 256, stride=2, pad=1),  # 8x8 -> 16x16
        (x, skip) -> begin
            # Center crop skip connection if needed
            skip = center_crop(skip, size(x)[1:2])
            cat(x, skip; dims=3)
        end,
        Conv((3,3), 512 => 256, pad=SamePad()),
        BatchNorm(256),
        x -> relu.(x)
    ) |> f32

    up2 = Chain(
        ConvTranspose((4,4), 256 => 128, stride=2, pad=1),  # 16x16 -> 32x32
        (x, skip) -> begin
            skip = center_crop(skip, size(x)[1:2])
            cat(x, skip; dims=3)
        end,
        Conv((3,3), 256 => 128, pad=SamePad()),
        BatchNorm(128),
        x -> relu.(x)
    ) |> f32

    # Final convolution
    final = Conv((1,1), 128 => out_ch) |> f32

    # Time embedding
    time_embed = Chain(
        Dense(1, time_dim),
        x -> relu.(x),
        Dense(time_dim, time_dim)
    ) |> f32

    return (x, t) -> begin
        t_emb = time_embed(reshape(t, 1, :))
        
        # Encoder
        x1 = conv0(x)        # 32x32
        x2 = down1(x1)       # 16x16
        x3 = down2(x2)       # 8x8
        
        # Bottleneck
        x4 = bottleneck(x3)  # 8x8
        
        # Decoder
        x_up = up1[1](x4)
        x_up = up1[2](x_up, x2)
        x_up = up1[3:end](x_up)
        
        x_up = up2[1](x_up)
        x_up = up2[2](x_up, x1)
        x_up = up2[3:end](x_up)
        
        return final(x_up)
    end
end

# Helper function for center cropping
function center_crop(x, new_size)
    h, w = size(x)[1:2]
    new_h, new_w = new_size
    h_start = div(h - new_h, 2) + 1
    w_start = div(w - new_w, 2) + 1
    return x[h_start:h_start+new_h-1, w_start:w_start+new_w-1, :, :]
end

"""
    get_data(dataset::Symbol, batch_size::Int)

Returns a Flux.DataLoader for the specified dataset.
`:mnist` loads the MNIST dataset.
`:synthetic` loads the 'SyntheticImages500.mat' dataset.
"""
function get_data(dataset::Symbol, batch_size::Int)
    if dataset == :mnist
        @info "Loading MNIST dataset..."
        xtrain, _ = MLDatasets.MNIST(:train)[:]
        # Pad MNIST images from 28x28 to 32x32 to match synthetic data
        xtrain_padded = padarray(reshape(xtrain, 28, 28, 1, :), Pad(:circular, (2, 2)))
        return Flux.DataLoader(xtrain_padded, batchsize=batch_size, shuffle=true)
    elseif dataset == :synthetic
        @info "Loading synthetic dataset..."
        data = matread(joinpath(@__DIR__, "..", "SyntheticImages500.mat"))
        raw = data["syntheticImages"]
        images = reshape(raw, 32, 32, 1, 500) # Add channel dimension
        return Flux.DataLoader(Float32.(images), batchsize=batch_size, shuffle=true)
    else
        throw(ArgumentError("Unsupported dataset: $dataset. Use :mnist or :synthetic."))
    end
end

"""
    train_model(; dataset::Symbol, epochs::Int, batch_size::Int, learning_rate::Float64, model_save_path::String="unet_model.bson")

Trains the U-Net model on the specified dataset.
"""
function train_model(; dataset::Symbol, epochs::Int=10, batch_size::Int=32, learning_rate::Float64=1e-4, model_save_path::String="unet_model.bson")

    loader = get_data(dataset, batch_size)
    loader = (Float32.(x) for x in loader)
    
    unet = build_unet() |> f32

    
    function loss(unet_model, x_clean)
        batch_size = size(x_clean, 4)
        t = rand(1:500, batch_size)
        noise = randn(Float32,size(x_clean))
        
        
        noisy_image = sqrt(0.8f0) .* x_clean .+ sqrt(0.2f0) .* noise 
        predicted_noise = unet_model(noisy_image, Float32.(t))
        return Flux.mse(predicted_noise, noise)
    end
    
    opt = ADAM(learning_rate)

   
    @info "Starting training on '$dataset' for $epochs epochs..."
    ps = Flux.params(unet)
    for epoch in 1:epochs
        @showprogress "Epoch $epoch " for x_clean in loader
            x_clean = Float32.(x_clean) 
            gs = gradient(() -> loss(unet, x_clean), ps)
            Flux.update!(opt, ps, gs)        
        end
        first_batch = first(loader)
        current_loss = loss(unet, first_batch)
        @info "Epoch: $epoch | Loss: $current_loss"
    end
    

    @info "Training complete. Saving model to $model_save_path"

    unet_cpu = cpu(unet)
    @save model_save_path unet_cpu
end

"""
    test_model(model_path::String="unet_model.bson"; output_filename::String="generated_image.png")

Loads a trained U-Net model and uses it to generate an image from pure noise.
"""
function test_model(model_path::String="unet_model.bson"; output_filename::String="generated_image.png", num_denoising_steps::Int=200)

    @info "Loading model from $model_path"
    @load model_path unet_cpu
    unet = unet_cpu 


    @info "Generating image from noise..."
    img = randn(Float32, 32, 32, 1, 1)
    

    @showprogress "Denoising " for t in reverse(1:num_denoising_steps)
        t_vec = fill(Float32(t), 1)  
        predicted_noise = unet(img, t_vec)

        img = (img .- sqrt(0.2f0) .* predicted_noise) ./ sqrt(0.8f0)
    end

    generated_img = clamp01.(img[:,:,1,1])
    save(output_filename, colorview(Gray, generated_img))
    @info "Generated image saved to $output_filename"
end


"""
    main()

Main function to run training and testing.
Parses command-line arguments to select the dataset.
"""
function main()

    dataset = :mnist 
    if "synthetic" in ARGS
        dataset = :synthetic
    end

    println("--- Starting Diffusion Model Pipeline ---")
    println("Selected Dataset: $dataset")
    

    train_model(dataset=dataset, epochs=20, batch_size=64, learning_rate=1e-3)
    

    test_model()

    println("--- Pipeline Finished ---")
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end  # End of module MyPackage
