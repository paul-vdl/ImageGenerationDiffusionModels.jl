using Flux: trainable, ADAM, setup, withgradient, update!
using BSON: @load
using Plots: heatmap
using ImageGenerationDiffusionModels: SimpleUnet

# Load the trained model
@load "trained_model.bson" model

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
    alpha_t = alpha_cum[t]
    alpha_prev = t > 1 ? alpha_cum[t_prev] : 1.0f0
    
    beta_t = 1f0 - alpha_t
    beta_prev = 1f0 - alpha_prev
    
    # Compute mean and variance
    sigma_t = sqrt(beta_t)
    
    # Reverse process mean prediction
    pred_x0 = (x_t .- sigma_t .* ϵ_pred) ./ sqrt(alpha_t)
    
    # Clamp to valid image range
    pred_x0 = clamp.(pred_x0, -1f0, 1f0)
    
    # Compute variance
    posterior_variance = (beta_prev * (1f0 - alpha_t)) / (1f0 - alpha_t)
    
    # Sample from the reverse process
    if t > 1
        noise = randn(Float32, size(x_t))
        x_prev = sqrt(alpha_prev) .* pred_x0 .+ 
                 sqrt(posterior_variance) .* noise
    else
        x_prev = pred_x0
    end
    
    return x_prev
end


# Image generation function
"""
    generate_image(;model=model, num_images=1, image_size=(32,32))

Generates images from noise using the given reverse diffusion model

# Arguments
- `model`: a trained U-Net model used to generate image
- `num_images`: number of images to be generated
- `image_size`: size of the image(s) to be generated

# Returns
- a 4-dimensional array of generated images
"""
function generate_image(;model=model, num_images=1, image_size=(32,32))
    # Start with pure noise
    x_t = randn(Float32, image_size..., 1, num_images)
    
    # Reverse diffusion process
    for t in reverse(2:T)
        t_prev = t - 1
        x_t = reverse_diffusion(model, x_t, t, t_prev)
    end
    
    # Final denoising step
    x_t = clamp.(x_t, -1f0, 1f0)

    # Visualize generated images
    for i in 1:num_images
        img = x_t[:,:,1,i]

        # Rescale from [-1,1] to [0,1] for display
        img_scaled = (img .+ 1f0) ./ 2f0
        
        # Save image
        heatmap(img_scaled, color=:grays, aspect_ratio=:equal, 
                title="Generated Image $i")
        savefig("generated_image_$i.png")    
    end
    
    return x_t
end

"""
    denoise_image(img; model=model)

Denoises an image using the given reverse diffusion model

# Arguments
- `img` : an image noised
- `model`: a trained U-Net model used to generate image
- `num_images`: number of images to be generated

# Returns
- a 4-dimensional array of the denoised image
"""

function denoise_image(img; model=model)
    # Adjust the image format
    x_t = convert(Array{Float32}, img)
    x_t = reshape(x_t, size(x_t, 1), size(x_t, 2), 1, 1)
    
    # Reverse diffusion process
    for t in reverse(2:T)
        t_prev = t - 1
        x_t = reverse_diffusion(model, x_t, t, t_prev)
    end
    
    # Final denoising step
    x_t = clamp.(x_t, -1f0, 1f0)

    # Visualize denoised image
    img = x_t[:,:,1,1]

    # Rescale from [-1,1] to [0,1] for display
    img_scaled = (img .+ 1f0) ./ 2f0
    
    # Save image
    heatmap(img_scaled, color=:grays, aspect_ratio=:equal, 
            title="Denoised Image")
    savefig("denoised_image.png")    
    
    return x_t
end
