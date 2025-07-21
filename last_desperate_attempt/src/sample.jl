#!/usr/bin/env julia
# Immediately parse your command‐line arguments
if length(ARGS) < 1
  println(stderr, "Usage: $(basename(PROGRAM_FILE)) <path‐to‐model.bson> [n_samples]")
  exit(1)
end

"""
    sample.jl

Command‐line sampling script for a trained diffusion model in Julia.

Usage:
  julia sample.jl <path‐to‐model.bson> [n_samples]

– Loads a `Shared.SimpleUNet` model plus normalization constants `μ, σ`  
  from the given BSON file.  
– Reconstructs the same σ‐schedule used in training (via `make_sigma_schedule`).  
– Generates `n_samples` images by running the reverse diffusion loop.  
– Displays them in a grid and writes `samples.png`.  

Requires:
  – Flux, BSON, ColorTypes, Plots  
  – `shared.jl` (which defines `SimpleUNet`).
"""

model_path = ARGS[1]

# optional second argument: number of samples
n = length(ARGS) >= 2 ? parse(Int32, ARGS[2]) : 16

using Flux
using BSON: @load
using ColorTypes
using Plots
include("shared.jl")   # defines SimpleUNet and make_sigma_schedule

# ############################################################## #
# (Re)construct exactly the same schedules you used for training #
# ############################################################## #
const T         = 750
const σ_min     = 1f-5
const σ_max     = 1.0f0
const ρ         = 7.0f0
function make_sigma_schedule(σ_min, σ_max, ρ, N)
  invρ = 1/ρ
  a = σ_min^(-invρ)
  b = σ_max^(-invρ)
  σ = zeros(Float64, N+1)
  σ[1] = 0.0
  for i in 2:N+1
    t = (i-2)/(N-1)
    σ[i] = min((a + t*(b - a))^(-ρ), 1)
  end
  σ
end
σ_schedule = Float32.(make_sigma_schedule(σ_min, σ_max, ρ, max(T-1, 3))[1:T])
#σ_schedule = Float32.(make_sigma_schedule(σ_min, σ_max, ρ, T-1))
#################################################################
@info "Rebuilt σ_schedule:"
#for t in 1:length(σ_schedule)
#  @info " t=$(t), σ=$(σ_schedule[t])"
#end

# Find any out‐of‐bounds entries
#bad_neg = findall(x -> x < 0f0, σ_schedule)
#bad_big = findall(x -> x >= 1f0, σ_schedule)
#bad_dec = findall(!issorted(σ_schedule), 1:length(σ_schedule)) 

#@info "Indices with σ < 0  : $bad_neg"
#@info "Indices with σ ≥ 1  : $bad_big"

# Check monotonicity more directly
for t in 2:length(σ_schedule)
  if σ_schedule[t] < σ_schedule[t-1]
    @warn "σ_schedule not non-decreasing at t=$(t): \
           σ[$(t-1)]=$(σ_schedule[t-1]) > σ[$t]=$(σ_schedule[t])"
  end
end


#@assert isempty(bad_neg) "σ_schedule has negative entries!"
#@assert isempty(bad_big) "σ_schedule has entries ≥ 1!"
# (monotonicity logged above)

#################################################################
@assert length(σ_schedule) == T
@assert all(σ_schedule .>= 0f0) && issorted(σ_schedule) && maximum(σ_schedule) <= 1f0

# —————————————————————————————————————————————————————————————
# Load trained model + normalizing μ, σ
# —————————————————————————————————————————————————————————————
model = Shared.SimpleUNet(1; base_ch=64, D_time=128)
@load model_path model μ σ
μ = Float32(μ)
σ = Float32(σ)

# Move model to GPU if you trained there
# model = gpu(model)
@info "Model loaded"
"""
    sample(model, μ, σ, n_samples; steps::Integer = T) -> Array{Float32,4}

Run reverse‐diffusion sampling on `n_samples` starting from pure Gaussian noise.

Arguments:
  • `model`          – a trained `Shared.SimpleUNet` that predicts noise given (x, t).  
  • `μ::Float32`     – per‐pixel mean used in the original data normaliziation.  
  • `σ::Float32`     – per‐pixel stddev used in the original data normalization.  
  • `n_samples::Int` – number of independent samples to generate.  
Keywords:
  • `steps::Int=T`   – number of diffusion timesteps (must match training T).

Returns:
  A 4‐dimensional `Float32` array of shape `(32,32,1,n_samples)` containing the
  final reconstructed images *after* re‐scaling by `σ` & `μ`, i.e.
  
      x_final = x_noise ⋅ σ .+ μ
"""
function sample(model, μ::Float32, σ::Float32, n_samples::Integer; steps::Integer=T)

  # Precompute α and sqrtα
  #alphas = @. 1f0 - σ_schedule^2
  #sqrt_alphas = sqrt.(alphas)


  # Start from pure Gaussian noise
  x = randn(Float32, 32,32,1,n_samples)
  @info "We have random noise …"
  @info "Sampling from T=$(steps) down to 1"
  for t in steps:-1:1
    # extra noise (Float32)
    z = t > 1 ? randn(Float32, size(x)) : zeros(Float32, size(x))

    # make a Float32 vector of timesteps
    ts = fill(Int32(t), n_samples)

    # predict the noise (model expects Float32 input)
    ϵ̂ = model(x, ts)
    @info "Noise prediction timetep $t"
    σt = σ_schedule[t]
    αt = 1f0 - σt^2
    @info "αt computed"
    sqrt_αt = max(sqrt(αt), 1f-7)
    #sqrtαt  = sqrt_alphas[t]
    #@as_sert all(isfinite, x)
    #@assert all(isfinite, σt)
    #@assert all(isfinite, ϵ̂)
    #@assert all(isfinite, sqrt_αt) #z
    @assert eltype(x) === Float32 "x pre"
    @assert eltype(σt) === Float32 "σt pre"
    @assert eltype(ϵ̂) === Float32 "ϵ̂ pre"
    @assert eltype(sqrt_αt) === Float32 "sqrt_αt pre"
    @assert eltype(z) === Float32 "z pre"
    # reverse‐diffusion step

    x = (x .- σt .* ϵ̂) ./ sqrt_αt .+ σt .* z
    @info "raw samples of timestep $t created"
    @assert eltype(x) === Float32
    @assert all(isfinite, x)
  end

  # undo the original data normalization
  return x .* σ .+ μ
end

# ##############
# Run sampling #
# ##############
n = 16
samples = sample(model, μ, σ, n)


@assert eltype(samples) === Float32

# visualize and save...
imgs = [Gray.(samples[:,:,1,i]) for i in 1:n]
cols = ceil(Int32, sqrt(n))
rows = ceil(Int32, n/cols)
@info "Plotting $n images in a $rows×$cols grid"
plt = plot(layout = (rows,cols),
           legend   = false,
           ticks    = false,
           framestyle = :none,
           size     = (cols*150, rows*150),
           suptitle="Our generated Images",
           aspect_ratio = 1,
           margin      = 2Plots.mm,
           dpi         = 150
          )

for i in 1:length(imgs)
  heatmap!(plt,
           imgs[i],
           c         = :grays, 
           colorbar  = false,
           subplot   = i
          )
end

display(plt)
savefig(plt, "samples.png")

@info "Saved generated samples to samples.png"
