module MyPackage

using MAT
using Images
using FileIO
using Flux
using NNlib: pad
using Statistics: mean
using MLDatasets

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
    variance_schedule = beta_min : 1 / num_noise_steps : beta_max #the way the model adds noise to the image
    epsilon = randn(size(img)) #gaussian noise

    for beta in variance_schedule 
    img = sqrt(1-beta) .* img + sqrt(beta) .* epsilon
    end
    
    canvas = clamp01.(img)
    save("noisy_img.png", colorview(Gray, canvas)) 
    return img
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


train_x, train_y = MNIST.traindata()
test_x,  test_y  = MNIST.testdata()

train_imgs = Float32.(train_x) ./ 255.0
train_imgs = reshape(train_imgs, 28, 28, 1, :)
function pad_to_32(imgs)
    padded = zeros(Float32, 32, 32, 1, size(imgs, 4))
    padded[3:30, 3:30, :, :] .= imgs
    return padded
end
train_imgs = pad_to_32(train_imgs)

num_steps = 500
beta_min, beta_max = 0.0001f0, 0.02f0
betas = collect(LinRange(beta_min, beta_max, num_steps))
alphas = 1 .- betas
alpha = accumulate(*, alphas)

function q_sample(x_0, t, epss, alpha)
    sqrt_alpha_t = reshape(sqrt.(alpha[t]), 1, 1, 1, :)
    sqrt_one_minus_alpha_t = reshape(sqrt.(1 .- alpha[t]), 1, 1, 1, :)
    return sqrt_alpha_t .* x_0 .+ sqrt_one_minus_alpha_t .* epss
end

function loss_fn(model, x_0, t, alpha)
    epss = randn(Float32, size(x_0))
    x_t = q_sample(x_0, t, epss, alpha)
    t_vec = reshape(Float32.(t), 1, size(x_0)[end])
    epss_pred = model(x_t, t_vec)
    return mean((epss_pred .- epss).^2)
end

train_loader = Flux.DataLoader(train_imgs, batchsize=128, shuffle=true)

model =build_unet()

optim = Flux.setup(Adam(3.0f-4), model)

losses = Float32[]

for epoch in 1:5
    for (i, x_0) in enumerate(train_loader)
        batch_size = size(x_0)[end]
        t = rand(1:num_steps, batch_size)
        loss, grads = Flux.withgradient(m -> loss_fn(m, x_0, t, alpha), model)
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)
        if isone(i) || iszero(i % 50)
            acc = accuracy(model) * 100
            @info "Epoch $epoch, step $i:\t loss = $(loss), acc = $(acc)%"
        end
    end
end

end
