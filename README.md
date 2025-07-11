# ImageGenerationDiffusionModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://paul-vdl.github.io/ImageGenerationDiffusionModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://paul-vdl.github.io/ImageGenerationDiffusionModels.jl/dev/)
[![Build Status](https://github.com/paul-vdl/ImageGenerationDiffusionModels.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/paul-vdl/ImageGenerationDiffusionModels.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/paul-vdl/ImageGenerationDiffusionModels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/paul-vdl/ImageGenerationDiffusionModels.jl)

The objective of this package is to generate images using a diffusion model.

We begin by progressively adding Gaussian noise to an input image. Then, we implement a U-Net architecture and train it to predict the noise that was added. This enables the model to learn how to reverse the diffusion process and reconstruct the original image.

Ultimately, the model becomes capable of generating a clean image from pure random noise by denoising it step by step—closely following the typical pipeline used in modern diffusion-based generative models.

## Available Functions

- `generate_grid()`  
  Imports the image `SyntheticImages500.mat` (original data) and converts it to a PNG file (`grid.png`). Also returns an array used by other functions.

- `apply_noise(img; num_noise_steps = 500, beta_min = 0.0001, beta_max = 0.02)`  
  Applies Gaussian noise to an image array gradually. Produces `noisy_img.png` and returns the noisy array.
  The default values for num_noise_steps, beta_min, and beta_max are based on commonly used settings in the diffusion model literature to ensure stable training denoising performance

- `train(data, lr::Float32=Float32(1e-4), epochs::Int=100, patience::Int=10, min_delta::Float64=0.001)`  
  Trains the model to map noisy → clean images.

- `denoise_image(noisy_img)`  
  De-noises a noisy image using the trained neural network.

- `generate_image(;model=model, num_images=1, image_size=(32,32))`  
  Generates a new image from random noise and de-noises it.

---

## Getting Started

Here's a quick example to get started:

In Pkg-mode:
```
add https://github.com/paul-vdl/ImageGenerationDiffusionModels.jl
```
Then in REPL:
```julia
using ImageGenerationDiffusionModels
```
```julia
demo()
```
It will call the other functions (except `train`) but you can also call them one by one:

```julia
img = generate_grid()
```
```julia
noisy_img = apply_noise(img)
```
```julia
denoised_img = denoise_image(img)
```
```julia
new_img = generate_image()
```
You can also train a model with a dataset (not recommended, it takes a lot of time)
```julia
train("SyntheticImages500.mat")
```
