#!/usr/bin/env julia
"""
    train.jl

Training script for a denoising diffusion UNet in Julia.

Usage:
  julia train.jl

This script will:
  1. Load and normalize 32×32 grayscale images from a `.mat` file.
  2. Build a linear β‐schedule → ᾱ → σ‐schedule for forward diffusion.
  3. Define a learning‐rate schedule with linear warmup + cosine decay.
  4. Train `Shared.SimpleUNet` with an MSE diffusion‐loss, using ADAMW + weight decay.
  5. Save checkpoints, early‐stopping models, and the final best model.
"""

using Flux, Flux.Optimise
using Flux.Losses: mse
using Statistics, Random
using BSON: @save
using MAT
include("shared.jl")

# ————————————————— hyper-params —————————————————
const DEBUG         = false
const T             = DEBUG ? 2  : 150
const BATCH_SIZE    = DEBUG ? 2  : 32
const EPOCHS        = DEBUG ? 5  : 100 #1
const PATIENCE      = DEBUG ? 2  : 9
const lr_start      = 3f-3 #5f-3 1f-2 2f-4
const lr_min        = 1f-6 # rsaise if we plateau 
const λ             = 1f-7 # weigth decay
const warmup_epochs = DEBUG ? 0 : 9
#TODO Make UNet wider (128 channel)or deeper (bigger base_ch, more layers, more time‐embedding dims). 64->128->256->512->1024.
# build β-schedule → ᾱ and σ-schedule
β_min, β_max = 1f-4, 2f-2
β = collect(range(β_min, β_max, length=T))
α = 1 .- β
α_bar = accumulate(*, α)
σ_schedule = sqrt.(1 .- α_bar)  # length T, Float32

"""
    get_lr(ep::Integer) -> Float32

Compute the current learning rate at epoch `ep`, using:
  • Linear warmup for the first `warmup_epochs`.
  • Constant max‐LR for the next `2*warmup_epochs` (if configured).
  • Cosine decay from `lr_start` down to `lr_min` thereafter.

Arguments:
  • `ep` — current epoch number (1‐based).

Returns:
  A `Float32` learning rate.
"""
function get_lr(ep) #longer at leak 
  if ep <= warmup_epochs
    return Float32(lr_start * ep/warmup_epochs)
  end
  if ep <= 3*warmup_epochs
    return lr_start
  else
    t = Float32((ep - warmup_epochs)/(EPOCHS - warmup_epochs))
    cosine = 0.5f0*(1 + cos(pi*t))
    return Float32(lr_min + (lr_start - lr_min)*cosine)
  end
end

#lr_max = lr_start
#function get_lr(ep)
#  t = (ep - 1) / (EPOCHS - 1)       
#  decay = 1f0 - abs(cos(pi * t))    # 0 -> 1 -> 0
#  return Float32(lr_min + (lr_max - lr_min) * decay)
#end


"""
    load_and_normalize(path::AbstractString) -> (imgs::Array{Float32,4}, μ::Float32, σ::Float32)

Read a MATLAB `.mat` file containing `syntheticImages` and:
  1. Convert to `Float32`.
  2. Reshape to `(H, W, C, N)` if necessary.
  3. Compute per‐pixel mean `μ` and standard deviation `σ`.
  4. Z‐score normalize all pixels: `(x - μ)/σ`.

Arguments:
  • `path` — filesystem path to the `.mat` file.

Returns:
  • `imgs` — normalized images as a 4D array `(H, W, C, N)`.  
  • `μ`    — original mean (Float32).  
  • `σ`    — original stddev (Float32).
"""
function load_and_normalize(path)
  mat = matread(path)
  raw = Float32.(mat["syntheticImages"])
  μ, σ = mean(raw), std(raw)
  raw .= (raw .- μ) ./ σ
  return raw, μ, σ
end


"""
    batch_iterator(imgs::Array{Float32,4}, bs::Integer) -> Channel{Array{Float32,4}}

Return an infinite channel producing random mini‐batches of size `bs`.

Arguments:
  • `imgs` — 4‐D array `(H, W, C, N)` of normalized images.  
  • `bs`   — batch size.

Behavior:
  • Randomly permutes the `N` images each epoch.  
  • Yields views of shape `(H, W, C, bs)` until exhausted, then reshuffles.  
  • Runs indefinitely until the channel is closed or collected.
"""
function batch_iterator(imgs, bs)
  N = size(imgs,4)
  Channel{Array{Float32,4}}(c->begin
    while true
      for perm in Iterators.repeated(randperm(N), 1)
        for i in 1:bs:N
          sel = perm[i:min(i+bs-1,N)]
          put!(c, view(imgs,:,:,:,sel))
        end
      end
    end
  end)
end

# forward‐diffusion loss
"""
    diffusion_loss(model, x0::Array{Float32,4}) -> Float32

Compute the denoising‐diffusion MSE loss for a batch `x0`.

Process:
  1. Sample random timesteps `tᵢ ∈ {1,…,T}` for each element in the batch.  
  2. Draw standard Gaussian noise `ε` of the same shape as `x0`.  
  3. Compute `xₜ = sqrt(1 - σ_t^2)*x0 + σ_t*ε` using the global `σ_schedule`.  
  4. Predict noise `ε̂ = model(xₜ, t)` and return `mse(ε̂, ε)`.

Arguments:
  • `model` — the UNet that maps `(x, timesteps)` → predicted noise.  
  • `x0`    — clean images batch `(H, W, C, B)`.

Returns:
  The MSE between predicted and true noise (scalar `Float32`).
"""
function diffusion_loss(model, x0)
  B = size(x0,4)
  ts = rand(1:T, B)
  ε = randn(Float32, size(x0))
  σvec = 0 .* σ_schedule[ts]
  σt   = reshape(σvec, (1,1,1,B)) 
  x_t = sqrt.(1 .- σt.^2) .* x0 .+ σt .* ε
  ϵ̂ = model(x_t, ts)
  return mse(ϵ̂, ε)
end

# training loop
"""
    train()

Execute the full training loop:

  • Loads and normalizes the dataset.
  • Instantiates `Shared.SimpleUNet`.
  • Sets up ADAMW optimizer with dynamic learning‐rate schedule and weight decay.
  • Iterates for `EPOCHS` epochs:
      – Adjusts LR via `get_lr`.
      – Loops over mini‐batches from `batch_iterator`.
      – Computes `diffusion_loss`, backpropagates, and updates model.
      – Logs and tracks best loss, with early stopping on `PATIENCE`.
      – Saves checkpoints every 10 epochs and the final model at end.

Outputs:
  – `best_model.bson` (best validation loss)  
  – `checkpoint_epochXX.bson` every 10 epochs  
  – `final_model.bson` on completion
"""
function train()
  imgs, μ, σ = load_and_normalize("data/SyntheticImages500.mat")
  batches = batch_iterator(imgs, BATCH_SIZE)
  _start_time = time()
  @info "Image data loaded and normalized"
  model = Shared.SimpleUNet(1; base_ch=80, D_time=128)
  opt = ADAMW(get_lr(1), (0.9f0, 0.999f0), λ)
  state = Flux.setup(opt, model)
  @info "Model instantiated"

  best = Inf; patience = 0

  @info "Elapsed=$(round(time() - _start_time, digits=1))s"
  @info "Starting the training …"
  #params = Flux.trainable(model)
  for epoch in 1:EPOCHS
    lr = get_lr(epoch)
    Flux.adjust!(state, eta=lr)
    total, nb = 0f0, 0
    for x0 in Iterators.take(batches, div(size(imgs,4),BATCH_SIZE))
      #loss, back = Flux.withgradient(() -> diffusion_loss(model,x0), ps)
      loss, grads = Flux.withgradient(model) do m 
            diffusion_loss(m,x0) #+ λ * sum(p->sum(abs2, p), Flux.trainable(m)) 
      end
      Flux.update!(state, model, grads[1])
      total += loss
      nb    += 1
      if DEBUG && nb ≥ 3 break end
    end #end batch loop

    train_loss = total/nb
    @info "Epoch $epoch — lr=$(round(lr,digits=4)) — loss=$(round(train_loss,digits=5))"

    if train_loss < best - 1e-8
      best = train_loss; patience = 0
      #@save "best_model.bson" model μ σ
    else
      patience += 1
      @info "  no improvement for $patience epochs"
      if patience ≥ PATIENCE
        @info "Early stopping."
        #@save "early_stopping_model.bson" model μ σ
        break
      end #endif
    end #endif

    @info "Epoch $epoch — elapsed=$(round(time() - _start_time, digits=1))s"

    if epoch % 10 == 0
      @save "checkpoint_epoch$(epoch).bson" model μ σ
    end #endif
  end #end training loop

  @info "Finished. best loss = $best"
  @save "final_model.bson" model μ σ
end #end function train()

train()

