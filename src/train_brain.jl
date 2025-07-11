using Flux: trainable, Chain, Conv, BatchNorm, MaxPool, ADAM, setup, withgradient, update!
using Zygote: gradient, @nograd
using Flux.Losses: mse 


using MAT: matread              # for .mat loading
using BSON: @save        # for checkpointing
using Plots: plot

# =================================================
# 1) Hyperparameters
# =================================================
const D = 128               # embedding dimension
const T = 500               # diffusion timesteps

const beta_min = Float32(1e-4)
const beta_max = Float32(0.02)
const beta     = collect(range(beta_min, beta_max, length=T))
const alpha     = 1 .- beta
const alpha_cum = accumulate(*, alpha)

const batch_size = 64
const epochs     = 100
const lr         = Float32(1e-4)

const patience = 10  # Number of epochs to wait for improvement
const min_delta = 0.001  # Minimum change to consider as improvement

# =================================================
# 2) Device (CPU‐only here)
# =================================================
device(x) = x

# =================================================
# 3) Sinusoidal timestep embedding
# =================================================

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
    pe[2*i-1] = sin(t * div)
    pe[2*i  ] = cos(t * div)
  end
  return pe
end
@nograd timestep_embedding

# =================================================
# 4) A small U-Net definition
# =================================================

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
    Conv((3,3), ch_out=>ch_out, pad=1), BatchNorm(ch_out), x->relu.(x)
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

    # reshape timestep embeddings into 1×1×D×B, then tile
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

# =================================================
# 5) Data loader
# =================================================

"""
    batch_iterator(imgs::Array{Float32,4}, bs::Int)


Provides an iterator over random mini batches of images   
# Arguments
- `imgs::Array{Float32,4}`: 4-dimensional array of images
- `bs::Int`: batch size

# Returns
- A `Channel` that yiels batches
"""
function batch_iterator(imgs::Array{Float32,4}, bs::Int)
  N = size(imgs,4)
  return Channel{Array{Float32,4}}(c -> begin
    idx = randperm(N)
    for i in 1:bs:N
      sel = idx[i:min(i+bs-1, N)]
      put!(c, device(view(imgs, :, :, :, sel)))
    end
  end)
end

# =================================================
# 6) Single-step loss (forward pass)
# =================================================

"""
    train_step(m::SimpleUNet, x0)

Performs a single training step for a denoising diffusion model using a Unet

# Arguments
- `m::SimpleUNet`: neural network model used to predict noise given a noisy embedding and a timestep embedding
- `x0`: A 4D array representing the original clean inputs

# Returns
- Mean-squared error (MSE) loss between the predicted noise and the true noise added to the input

"""
function train_step(m::SimpleUNet, x0)
    B  = size(x0,4)
    ts = rand(1:T, B)                     # random timesteps per example
    ϵ  = randn(Float32, size(x0))        # noise

    alphas = alpha_cum[ts]
    a  = reshape(sqrt.(alphas), 1,1,1,B)
    b  = reshape(sqrt.(1 .- alphas), 1,1,1,B)
    x_t = a .* x0 .+ b .* ϵ               # noisy input

    t_emb  = hcat(timestep_embedding.(ts)...) |> device
    t_emb = reshape(t_emb, 1, 1, D, B)  # Ensure correct shape
    
    ϵ_pred = m((x_t, t_emb))

    return mse(ϵ_pred, ϵ)
end

# =================================================
# 8) Main training loop
# =================================================
function train()
    # Load & prepare data
    mat  = matread("SyntheticImages500.mat")
    raw  = mat["syntheticImages"]       # size (32,32,500)
    imgs = reshape(Float32.(raw), 32,32,1,:)
    imgs .*= 2; imgs .-= 1              # scale to [-1,1]

    # Create model & optimizer
    model = SimpleUNet(1)
    opt   = Adam(lr)
    # Set the optimizer state up
    state = setup(opt, model)
    # Train
    # Add loss tracking
    losses = Float32[]
    best_loss = Inf
    epochs_no_improve = 0
    for epoch in 1:epochs
        total_loss, n = Float32(0), 0
        for x0 in batch_iterator(imgs, batch_size)
            # Compute loss and gradients
            loss, grads = withgradient(model) do m
                train_step(m, x0)
            end
            
            # Update model parameters
            update!(state, model, grads[1])
            
            total_loss += loss
            n += 1
        end
        epoch_loss = total_loss/n
        push!(losses, epoch_loss)

        @info "Epoch $epoch | avg loss = $epoch_loss"
        # Early stopping logic
        if epoch_loss < best_loss - min_delta
            best_loss = epoch_loss
            epochs_no_improve = 0
        else
            epochs_no_improve += 1
        end

        # Early stopping condition
        if epochs_no_improve > patience
            @warn "Early stopping: No significant improvement for $(patience+1) epochs"
            break
        end
    end

    @save "trained_model.bson" model opt
    @info "Training complete! Model saved."
    plot(losses, title="Training Loss", xlabel="Epoch", ylabel="Loss")
    savefig("training_loss.png")
end
