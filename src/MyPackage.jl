module MyPackage

using MAT
using Flux
using Images
using FileIO
using Statistics
using Random
using Flux: mse, DataLoader, params, ADAM, gradient, update!
using Printf

function load_synthetic_images(path="SyntheticImages500.mat")
    data = matread(path)
    raw = data["syntheticImages"]
    Float32.(reshape(raw, 32, 32, 1, :) ./ 255)
end

function sinusoidal_embedding(t::Vector{Float32}, dim::Int)
    half_dim = div(dim, 2)
    emb = log(10000.0f0) / (half_dim - 1)
    freqs = Float32.(exp.(-emb .* (0:half_dim - 1)))
    angles = t .* freqs'
    Float32.(hcat(sin.(angles), cos.(angles)))
end

struct UNet
    conv0
    down1
    down2
    up1
    up2
    final
    time_mlp
end

Flux.@functor UNet

function (m::UNet)(x, t_vec)
    bsz = size(x)[end]
    time_dim = size(m.time_mlp.weight, 2)
    t_emb = sinusoidal_embedding(t_vec, time_dim)'
    t_encoded = relu(m.time_mlp(t_emb))
    t_proj = reshape(t_encoded, 1, 1, :, bsz)
    t_proj = repeat(t_proj, 8, 8, 1, 1)

    x1 = relu(m.conv0(x))
    x2 = m.down1(x1)
    x3 = m.down2(x2)
    x3 = x3 .+ t_proj  # Non-mutating addition to avoid AD errors
    x = m.up1(x3)
    x = m.up2(x)
    m.final(x)
end

function build_unet(in_ch=1, out_ch=1, time_dim=128)
    UNet(
        Conv((3,3), in_ch=>64, pad=1),
        Chain(Conv((3,3), 64=>128, pad=1, stride=2), relu),
        Chain(Conv((3,3), 128=>256, pad=1, stride=2), relu),
        Chain(ConvTranspose((3,3), 256=>128, pad=1, stride=2, outpad=1), relu),
        Chain(ConvTranspose((3,3), 128=>64, pad=1, stride=2, outpad=1), relu),
        Conv((1,1), 64=>out_ch),
        Dense(time_dim, 256)
    )
end

function apply_noise(img; beta_min=0.0001, beta_max=0.02, steps=500)
    t = rand(1:steps)
    beta = beta_min + (beta_max - beta_min) * t / steps
    ε = randn(Float32, size(img))
    noisy = sqrt(1 - beta) * img .+ sqrt(beta) * ε
    noisy, ε, Float32(t) / steps
end

loss_fn(model, x, t, noise) = mse(model(x, t), noise)

function train_model!(model, images; batch_size=32, epochs=3, η=1e-3)
    opt = ADAM(η)
    ps = params(model)
    state = Flux.setup(opt, ps)
    loader = DataLoader(images, batchsize=batch_size, shuffle=true)

    for epoch in 1:epochs
        total_loss = 0f0
        n = 0

        for batch in loader
            bsz = size(batch)[end]

            noisy_list = Vector{Array{Float32,3}}(undef, bsz)
            noise_list = Vector{Array{Float32,3}}(undef, bsz)
            t_vec = zeros(Float32, bsz)

            for i in 1:bsz
                noisy_i, noise_i, t_i = apply_noise(batch[:,:,:,i])
                noisy_list[i] = noisy_i
                noise_list[i] = noise_i
                t_vec[i] = t_i
            end

            noisy_batch = cat(noisy_list..., dims=4)
            noise_batch = cat(noise_list..., dims=4)

            grads = gradient(() -> loss_fn(model, noisy_batch, t_vec, noise_batch), ps)
            update!(state, ps, grads)
            total_loss += loss_fn(model, noisy_batch, t_vec, noise_batch)
            n += 1
        end

        @printf("Epoch %d | Loss = %.6f\n", epoch, total_loss / n)
    end
end

function sample_image(model, img_shape=(32,32,1); steps=500, beta_min=0.0001, beta_max=0.02)
    x = randn(Float32, img_shape...)
    for i in steps:-1:1
        t = Float32(i) / steps
        beta = beta_min + (beta_max - beta_min) * i / steps
        pred = model(reshape(x, 32, 32, 1, 1), [t])[:, :, :, 1]
        x = (x .- sqrt(beta) * pred) ./ sqrt(1 - beta)
    end
    clamp01.(x)
end

function run_training()
    images = load_synthetic_images()
    model = build_unet()

    # Initialize all parameters by running a dummy forward pass
    dummy_t = fill(0.0f0, 1)        # batch size 1 time scalar
    dummy_x = rand(Float32, 32, 32, 1, 1)
    model(dummy_x, dummy_t)

    println("[INFO] Training start...")
    train_model!(model, images)
    println("[INFO] Sampling...")
    img = sample_image(model)
    save("sample.png", colorview(Gray, img))
end

end








