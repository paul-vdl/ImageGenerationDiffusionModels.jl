# ImageGenerationDiffusionModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://paul-vdl.github.io/ImageGenerationDiffusionModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://paul-vdl.github.io/ImageGenerationDiffusionModels.jl/dev/)
[![Build Status](https://github.com/paul-vdl/ImageGenerationDiffusionModels.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/paul-vdl/ImageGenerationDiffusionModels.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/paul-vdl/ImageGenerationDiffusionModels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/paul-vdl/ImageGenerationDiffusionModels.jl)

The objective of this package is to generate an image with a diffusion model.  
We first train a diffusion network to add noise to an existing image and then train it to remove the noise.  
The model learns to generate a clear image from random noise.

---

## Available Functions

- `ImageGenerationDiffusionModels.generate_grid()`  
  Imports the image `SyntheticImages500.mat` and converts it to a PNG file (`grid.png`). Also returns an array used by other functions.

- `ImageGenerationDiffusionModels.apply_noise(img; num_noise_steps = 500, beta_min = 0.0001, beta_max = 0.02)`  
  Applies Gaussian noise to an image array gradually. Produces `noisy_img.png` and returns the noisy array.

- `ImageGenerationDiffusionModels.train_brain(num_steps::Int = 500)`  
  Trains the model to map noisy → clean images.

- `ImageGenerationDiffusionModels.denoise_image(noisy_img::AbstractMatrix{<:Real})`  
  De-noises a noisy image using the trained neural network.

- `ImageGenerationDiffusionModels.generate_image_from_noise()`  
  Generates a new image from random noise and de-noises it.

---

## Getting Started

Here's a quick example to get started:

```julia
using ImageGenerationDiffusionModels

img = ImageGenerationDiffusionModels.generate_grid()
noisy_img = ImageGenerationDiffusionModels.apply_noise(img)
ImageGenerationDiffusionModels.train_brain()
denoised_img = ImageGenerationDiffusionModels.denoise_image(img[1:32, 1:32])
new_img = ImageGenerationDiffusionModels.generate_image_from_noise()
