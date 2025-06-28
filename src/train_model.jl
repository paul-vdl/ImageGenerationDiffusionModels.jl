include("MyPackage.jl")
using .MyPackage

using Flux
using MLDatasets

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

model = MyPackage.build_unet()

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

