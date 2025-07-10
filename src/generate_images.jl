#!/usr/bin/env julia
using Flux
using BSON: @load
using Random
using Plots

# Device handling (CPU-only in this case)
device(x) = x

# Load the same constants from the training script
const D = 128               # embedding dimension
const T = 5 #00               # diffusion timesteps

const β_min = Float32(1e-4)
const β_max = Float32(0.02)
const β     = collect(range(β_min, β_max, length=T))
const α     = 1 .- β
const α_cum = accumulate(*, α)    # ᾱ_t = ∏ₛ αₛ

# Copy the SimpleUNet struct definition from the training script

"""
    ConvBlock(ch_in, ch_out)

Constructs a convolutional block consisting of two sequential convolutional layers

# Arguments
- `ch_in::Int`: number of input channels
- `ch_out::Int`: number of output channels

# Returns 
- Convolutional block
"""
function ConvBlock(ch_in, ch_out)
  Chain(
    Conv((3,3), ch_in=>ch_out, pad=1), BatchNorm(ch_out), x->relu.(x),
    Conv((3,3), ch_out=>ch_out, pad=1), BatchNorm(ch_out), x->relu.(x),
  )
end

struct SimpleUNet
    down1::Chain
    down2::Chain
    mid::Chain
    up2::Chain
    up1::Chain
    final::Conv
end

"""
    SimpleUNet(channels::Int=1) 

Constructs a simple U-Net model

# Arguments
- `channels::Int`: number of input channels

# Returns 
- `SimpleUNet :: Struct`
"""
function SimpleUNet(channels::Int=1)
    down1 = Chain(
        Conv((3,3), channels + D => 64, pad=1),
        BatchNorm(64, relu),
        Conv((3,3), 64 => 64, pad=1),
        BatchNorm(64, relu)
    )
    down2 = Chain(
        MaxPool((2,2)),
        Conv((3,3), 64 => 128, pad=1),
        BatchNorm(128, relu),
        Conv((3,3), 128 => 128, pad=1),
        BatchNorm(128, relu)
    )
    mid = Chain(
        Conv((3,3), 128 => 128, pad=1),
        BatchNorm(128, relu),
        Conv((3,3), 128 => 128, pad=1),
        BatchNorm(128, relu)
    )
    up2 = Chain(
        ConvTranspose((2,2), 128 => 64, stride=2),
        Conv((3,3), 64 => 64, pad=1),
        BatchNorm(64, relu),
        Conv((3,3), 64 => 64, pad=1),
        BatchNorm(64, relu)
    )
    up1 = Chain(
        Conv((3,3), 128 => 64, pad=1),
        BatchNorm(64, relu),
        Conv((3,3), 64 => 64, pad=1),
        BatchNorm(64, relu)
    )
    final = Conv((1,1), 64 => 1)
    
    SimpleUNet(down1, down2, mid, up2, up1, final)
end

"""

    (m::SimpleUNet)(x_and_emb)

Applies the forward pass of the 'SimpleUNet' to a noisy image

# Arguments
- `x_and_emb`: a tuple containg the input image batch and a timestep embedding

# Returns
- A 4-dimensional array
"""
function (m::SimpleUNet)(x_and_emb)
    x, t_emb = x_and_emb
    B = size(x,4)

    # reshape timestep embeddings into 1×1×D×B then tile
    tmap = reshape(t_emb, 1,1,:,B)  
    tmap = repeat(tmap, size(x,1), size(x,2), 1, 1)

    # forward U-Net pass with skip connections
    h1 = m.down1(cat(x, tmap; dims=3))
    h2 = m.down2(h1)
    h3 = m.mid(h2)
    up_h3 = m.up2(h3)

    H, W = size(up_h3,1), size(up_h3,2)
    h1_c = h1[1:H, 1:W, :, :]
    cat_feat = cat(up_h3, h1_c; dims=3)

    up = m.up1(cat_feat)
    return m.final(up)
end

# Timestep embedding function (same as in training script)

"""
    timestep_embedding(t::Integer; D::Int=D)

Generates a sinusoidal timestep embedding of dimension D

# Arguments:
- `t::Int`: Timestep index
- `D::Int`: Dimensioin of the embedding

# Returns
- A vector of length D containg sinusoidal embeddings
"""
function timestep_embedding(t::Integer; D::Int=D)
  pe = zeros(Float32, D)
  for i in 1:(D ÷ 2)
    div = exp(-log(Float32(1e4)) * (2*(i-1)/(D-1)))
    pe[2i-1] = sin(t * div)
    pe[2i  ] = cos(t * div)
  end
  return pe
end


# Reverse diffusion process

"""
    reverse_diffusion(model, x_t, t, t_prev)

Performs a single reverse diffusion step using the model

# Arguments
- `model`: a neural network that predicts the noise component
- `x_t::Array{Float32, 4}`: noisy image at timestep 't'
- `t::Int`: Current diffusion timestep
- `t_prev::Int`: previous timestep to step forward

# Returns
- `x_{t-1}`: estimated image at timestep `t_prev`
"""
function reverse_diffusion(model, x_t, t, t_prev)
    # Prepare timestep embedding of shape (1,1,D,B)
    B = size(x_t, 4)                       # batch size
    v = timestep_embedding(t)              # (D,)
    t_emb = reshape(v, 1, 1, length(v), 1) # (1,1,D,1)
    t_emb = repeat(t_emb, 1, 1, 1, B)      # (1,1,D,B)
    t_emb = device(t_emb)

    # Predict noise
    ϵ_pred = model((x_t, t_emb))
    
    # Compute coefficients
    α_t = α_cum[t]
    α_prev = t > 1 ? α_cum[t_prev] : 1.0f0
    
    β_t = 1f0 - α_t
    β_prev = 1f0 - α_prev
    
    # Compute mean and variance
    σ_t = sqrt(β_t)
    
    # Reverse process mean prediction
    pred_x0 = (x_t .- σ_t .* ϵ_pred) ./ sqrt(α_t)
    
    # Clamp to valid image range
    pred_x0 = clamp.(pred_x0, -1f0, 1f0)
    
    # Compute variance
    posterior_variance = (β_prev * (1f0 - α_t)) / (1f0 - α_t)
    
    # Sample from the reverse process
    if t > 1
        noise = randn(Float32, size(x_t))
        x_prev = sqrt(α_prev) .* pred_x0 .+ 
                 sqrt(posterior_variance) .* noise
    else
        x_prev = pred_x0
    end
    
    return x_prev
end


# Image generation function
"""
    generate_image(model; num_images=1, image_size=(32,32))

Generates images from noise using the given reverse diffusion model

# Arguments
- `model`: a trained U-Net model used to generate image
- `num_images`: number of images to be generated
- `image_size`: size of the image(s) to be generated

# Returns
- a 4-dimensional array of generated images
"""
function generate_image(model; num_images=1, image_size=(32,32))
    # Start with pure noise
    x_t = randn(Float32, image_size..., 1, num_images)
    
    # Reverse diffusion process
    for t in reverse(2:T)
        t_prev = t - 1
        x_t = reverse_diffusion(model, x_t, t, t_prev)
    end
    
    # Final denoising step
    x_t = clamp.(x_t, -1f0, 1f0)
    
    return x_t
end

# Main generation script
function main()
    # Load the trained model
    @load "trained_model.bson" model
    
    # Generate images
    generated_images = generate_image(model, num_images=5)
    
    # Visualize generated images
    for i in 1:size(generated_images, 4)
        img = generated_images[:,:,1,i]
        # Rescale from [-1,1] to [0,1] for display
        img_scaled = (img .+ 1f0) ./ 2f0
        
        # Save image
        heatmap(img_scaled, color=:grays, aspect_ratio=:equal, 
                title="Generated Image $i")
        savefig("generated_image_$i.png")
    end
    
    println("Image generation complete!")
end

# Run the generation
main()

