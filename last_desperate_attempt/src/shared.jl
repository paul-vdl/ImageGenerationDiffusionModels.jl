"""
# Shared

This module implements a Simple U-Net architecture designed for diffusion-based image generation tasks. The U-Net model incorporates sinusoidal timestep embeddings to effectively process temporal information during the image generation process. 

## Components
- **Timestep Embedding**: Generates sinusoidal embeddings based on discrete time steps.
- **TimeEmbedMLP**: A small multi-layer perceptron that projects time embeddings to a fixed output dimension.
- **add_time!**: A helper function that adds time features to the model's activations using a dense layer.
- **SimpleUNet**: The main U-Net structure, consisting of downsampling and upsampling paths, with integrated time embeddings for enhanced performance in diffusion models.

This architecture is particularly suited for tasks involving generative models, where the temporal aspect of the data plays a crucial role in the generation process.
"""
module Shared

using Flux
using Flux: Conv, GroupNorm, MaxPool, Upsample, sigmoid, cat, swish
import Flux: @functor
using Functors
using Zygote
using Statistics

export SimpleUNet, timestep_embedding

# ##############################################
# 1) Sinusoidal timestep embedding             #
# ##############################################
const D = 128

"""
    timestep_embedding(t::Integer; D::Int=128)

Generates a sinusoidal timestep embedding for a given integer `t`. The embedding is a vector of length `D`, where the first half contains sine values and the second half contains cosine values, computed based on the input `t`.

# Arguments
- `t::Integer`: The timestep for which the embedding is generated.
- `D::Int`: The dimensionality of the embedding (default is 128).

# Returns
A vector of type `Float32` of length `D` representing the sinusoidal embedding.
"""
function timestep_embedding(t::Integer; D::Int=D)
  pe = zeros(Float32, D)
  for i in 1:(D ÷ 2)
    div = exp(-log(1f4) * (2*(i-1)/(D-1)))
    pe[2i-1] = sin(t * div)
    pe[2i]   = cos(t * div)
  end
  return pe
end
Zygote.@nograd timestep_embedding

# ##############################################
# 2) A small MLP: D_time → 128                 #
# ##############################################
struct TimeEmbedMLP
  proj1::Dense   # D_time → hidden
  proj2::Dense   # hidden → outdim (128)
end

"""
    TimeEmbedMLP(D_time::Int; hidden::Int=256, outdim::Int=128)

Constructs a small multi-layer perceptron (MLP) for time embedding. 
The MLP projects an input of dimension `D_time` to a hidden layer and then to an output dimension of 128.

# Arguments
- `D_time::Int`: The input dimension for the MLP.
- `hidden::Int`: The number of neurons in the hidden layer (default is 256).
- `outdim::Int`: The output dimension of the MLP (default is 128).

# Returns
An instance of `TimeEmbedMLP` containing two dense layers.
"""
function TimeEmbedMLP(D_time::Int; hidden::Int=256, outdim::Int=128)
  TimeEmbedMLP(
    Dense(D_time, hidden, swish),
    Dense(hidden, outdim)   # no activation on last layer
  )
end

# apply MLP to one embedding vector
(m::TimeEmbedMLP)(x::AbstractVector{Float32}) = m.proj2(m.proj1(x))
Zygote.@nograd TimeEmbedMLP

# ##############################################
# 3) add_time! helper (batched broadcast)      #
# ##############################################
"""
    add_time!(h::Array{Float32,4}, bias_layer::Dense, tfeat::Array{Float32,2})

Adds time features to a 4D array `h` using a bias layer. The function computes a bias matrix from the time features and adds it to the input array.

# Arguments
- `h::Array{Float32,4}`: A 4D array of shape (H, W, C, B) representing the input features.
- `bias_layer::Dense`: A dense layer used to compute the bias from the time features.
- `tfeat::Array{Float32,2}`: A 2D array of shape (D_in, B) representing the time features.

# Returns
A new 4D array of the same shape as `h`, with the time features added.
"""
function add_time!(h::Array{Float32,4}, bias_layer::Dense, tfeat::Array{Float32,2})
  # h     :: H×W×C×B
  # tfeat :: D_in×B
  H, W, C, B = size(h)
  #@assert size(tfeat) == (C, B)
  D_in, B2  = size(tfeat)
  @assert B2 == B "batch‐size mismatch: got tfeat size $(size(tfeat)), but h has batch $B"
  Wmat = bias_layer.weight      # Flux ≥ 0.12: weight is (out, in)
  out, inn = size(Wmat)
  @assert inn == D_in "Dense input dim $inn != size(tfeat,1)=$D_in"
  @assert out == C     "Dense output dim $out != channel count C=$C"

  # Compute the C×B bias matrix (tfeat columns thu Dense , C vec)
  bcols = bias_layer.(eachcol(tfeat))  # B‐element Vector{Vector{Float32}} each of length C
  b4     = reshape(hcat(bcols...), 1,1,C,B)  # shape (1,1,C,B)

  # Return a new array instead of mutating in place
  return h .+ b4
end




# ##############################################
# 4) The 4-level Simple U-Net                  #
# ##############################################
struct SimpleUNet
  # time embedding
  time_mlp        :: TimeEmbedMLP

  # down path
  down1_conv      :: Chain
  down1_timebias  :: Dense
  down1_pool      :: MaxPool

  down2_conv      :: Chain
  down2_timebias  :: Dense
  down2_pool      :: MaxPool

  down3_conv      :: Chain
  down3_timebias  :: Dense
  down3_pool      :: MaxPool

  down4_conv      :: Chain
  down4_timebias  :: Dense
  down4_pool      :: MaxPool

  # bottleneck
  mid_conv        :: Chain
  mid_timebias    :: Dense

  # up path
  up3_upsample    :: Upsample
  up3_conv        :: Chain
  up3_timebias    :: Dense

  up2_upsample    :: Upsample
  up2_conv        :: Chain
  up2_timebias    :: Dense

  up1_upsample    :: Upsample
  up1_conv        :: Chain
  up1_timebias    :: Dense

  up0_upsample    :: Upsample
  up0_conv        :: Chain
  up0_timebias    :: Dense

  # final projection
  final_conv      :: Chain
end
@functor SimpleUNet

"""
    SimpleUNet(in_ch::Int=1; base_ch::Int=64, D_time::Int=128)

Constructs a 4-level Simple U-Net model for image processing tasks. The model includes downsampling and upsampling paths, with time embeddings integrated into the architecture.

# Arguments
- `in_ch::Int`: The number of input channels (default is 1).
- `base_ch::Int`: The base number of channels for the first layer (default is 64).
- `D_time::Int`: The dimensionality of the time embedding (default is 128).

# Returns
An instance of `SimpleUNet` configured with the specified parameters.
"""
function SimpleUNet(in_ch::Int=1; base_ch::Int=64, D_time::Int=128)
  # time MLP: D_time → 128
  t_mlp = TimeEmbedMLP(D_time; hidden=256, outdim=128)

  # helper conv block
  conv_block(inC, outC) = Chain(
    Conv((3,3), inC=>outC, pad=1),
    Flux.GroupNorm(outC, 8), swish,
    Conv((3,3), outC=>outC, pad=1),
    Flux.GroupNorm(outC, 8), swish
  )

  # aliases for readability
  b1, b2, b4, b8, b16 = base_ch, 2base_ch, 4base_ch, 8base_ch, 16base_ch

  return SimpleUNet(
    # time embed
    t_mlp,

    # down1: in_ch -> base_ch
    conv_block(in_ch, b1),
    Dense(128, b1),
    MaxPool((2,2)),

    # down2: base_ch -> 2*base_ch
    conv_block(b1, b2),
    Dense(128, b2),
    MaxPool((2,2)),

    # down3: 2*base_ch -> 4*base_ch
    conv_block(b2, b4),
    Dense(128, b4),
    MaxPool((2,2)),

    # down4: 4*base_ch -> 8*base_ch
    conv_block(b4, b8),
    Dense(128, b8),
    MaxPool((2,2)),

    # bottleneck: 8*base_ch -> 16*base_ch
    Chain(
      Conv((3,3), b8=>b16, pad=1), Flux.GroupNorm(b16, 8), swish,
      Conv((3,3), b16=>b16, pad=1), Flux.GroupNorm(b16, 8), swish
    ),
    Dense(128, b16),

    # up3: [16] [8], concat skip of [8] -> conv to [8]
    Upsample((2,2), :bilinear),
    Chain(
      Conv((3,3),(b16 + b8)=>b8, pad=1),
      Flux.GroupNorm(b8, 8), swish,
      Conv((3,3), b8=>b8, pad=1),
      Flux.GroupNorm(b8, 8), swish
    ),
    Dense(128, b8),

    # up2: [8]  [4], concat skip of [4] -> conv to [4]
    Upsample((2,2), :bilinear),
    Chain(
      Conv((3,3),(b8 + b4)=>b4, pad=1),
      Flux.GroupNorm(b4, 8), swish,
      Conv((3,3), b4=>b4, pad=1),
      Flux.GroupNorm(b4, 8), swish
    ),
    Dense(128, b4),

    # up1: [4] [2], concat skip of [2] -> conv to [2]
    Upsample((2,2), :bilinear),
    Chain(
      Conv((3,3),(b4 + b2)=>b2, pad=1),
      Flux.GroupNorm(b2, 8), swish,
      Conv((3,3), b2=>b2, pad=1),
      Flux.GroupNorm(b2, 8), swish
    ),
    Dense(128, b2),

    # up0: [2] [1], concat skip of [1] -> conv to [1]
    Upsample((2,2), :bilinear),
    Chain(
      Conv((3,3),(b2 + b1)=>b1, pad=1),
      Flux.GroupNorm(b1, 8), swish
    ),
    Dense(128, b1),

    # final conv -> 1 channel + sigmoid
    Chain(
      Conv((1,1), b1=>1, bias=false),
      sigmoid
    )
  )
end

# Forward-pass
"""
    (u::SimpleUNet)(x::Array{Float32,4}, ts::Vector{Integer})

Performs a forward pass through the Simple U-Net model. 
The function takes an input tensor and a vector of timesteps, applying the model's layers sequentially.

# Arguments
- `u::SimpleUNet`: The Simple U-Net instance.
- `x::Array{Float32,4}`: A 4D input tensor of shape (H, W, 1, B).
- `ts::Vector{Integer}`: A vector of timesteps corresponding to the batch.

# Returns
A 4D output tensor of shape (H, W, 1, B) representing the model's predictions.
"""
function (u::SimpleUNet)(x::Array{Float32,4}, ts::Vector{Integer})
  # x: H×W×1×B, ts: length-B
  H, W, _, B = size(x)

  # 1) Time embeddings
  raw_emb = hcat(timestep_embedding.(ts)...)       # (D_time, B)
  tcols   = u.time_mlp.(eachcol(raw_emb))         # Vector of B vectors (size 128)
  tfeat   = hcat(tcols...)                        # (128, B)

  # DOWN 1
  e1  = u.down1_conv(x)                            # H×W×b1×B
  e1  = add_time!(e1, u.down1_timebias, tfeat)
  e1p = u.down1_pool(e1)                           # (H/2)×(W/2)×b1×B

  # DOWN 2
  e2  = u.down2_conv(e1p)
  e2  = add_time!(e2, u.down2_timebias, tfeat)
  e2p = u.down2_pool(e2)                           # (H/4)

  # DOWN 3
  e3  = u.down3_conv(e2p)
  e3  = add_time!(e3, u.down3_timebias, tfeat)
  e3p = u.down3_pool(e3)                           # (H/8)

  # DOWN 4
  e4  = u.down4_conv(e3p)
  e4  = add_time!(e4, u.down4_timebias, tfeat)
  e4p = u.down4_pool(e4)                           # (H/16)

  # BOTTLENECK
  b   = u.mid_conv(e4p)
  b   = add_time!(b, u.mid_timebias, tfeat)         # (H/16)

  # UP 3
  u3  = u.up3_upsample(b)                           # (H/8)
  u3  = cat(u3, e4; dims=3)                         # concat channels
  u3  = u.up3_conv(u3)
  u3  = add_time!(u3, u.up3_timebias, tfeat)

  # UP 2
  u2  = u.up2_upsample(u3)                          # (H/4)
  u2  = cat(u2, e3; dims=3)
  u2  = u.up2_conv(u2)
  u2  = add_time!(u2, u.up2_timebias, tfeat)

  # UP 1
  u1 = u.up1_upsample(u2)                          # (H/2)
  u1 = cat(u1, e2; dims=3)
  u1 = u.up1_conv(u1)
  u1 = add_time!(u1, u.up1_timebias, tfeat)

  # UP 0
  u0 = u.up0_upsample(u1)                          # (H)
  u0 = cat(u0, e1; dims=3)
  u0 = u.up0_conv(u0)
  u0 = add_time!(u0, u.up0_timebias, tfeat)

  # FINAL
  return u.final_conv(u0)                          # H×W×1×B
end


end # module Shared
