# ImageGenerationDiffusionModels

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://paul-vdl.github.io/ImageGenerationDiffusionModels.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://paul-vdl.github.io/ImageGenerationDiffusionModels.jl/dev/)
[![Build Status](https://github.com/paul-vdl/ImageGenerationDiffusionModels.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/paul-vdl/ImageGenerationDiffusionModels.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/paul-vdl/ImageGenerationDiffusionModels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/paul-vdl/ImageGenerationDiffusionModels.jl)

The objective of this package is to generate an image with a diffusion model. In order to do this, we first train a diffusion network to add noise to an existing image and then, we will train it to remove the noise. Thus, the diffusion network will be able to generate a clear image from a random noise image.

For now, the module has five functions : ImageGenerationDiffusionModels.generate_grid, ImageGenerationDiffusionModels.apply_noise, ImageGenerationDiffusionModels.train_brain, ImageGenerationDiffusionModels.denoise_image, and ImageGenerationDiffusionModels.generate_image_from_noise. 

ImageGenerationDiffusionModels.generate_grid() : imports the image SyntheticImages500.mat and transforms it into a png image named grid.png that you can see in your repository and an array that could be used for the other functions.

ImageGenerationDiffusionModels.apply_noise(img; num_noise_steps = 500, beta_min = 0.0001, beta_max = 0.02) : take an image as an array and apply gaussian noise to it gradually. It generates a png image named "noisy_img.png" in your repository and an array that could be used for the other functions.

ImageGenerationDiffusionModels.train_brain(num_steps::Int=500) : trains the network to map noisy→clean.

ImageGenerationDiffusionModels.denoise_image(noisy_img::AbstractMatrix{<:Real}) : de-noises a noisy image using the trained neural network. Given a single input `noisy_img::Matrix{<:Real}`, this function produces a de-noised version of that input file.

ImageGenerationDiffusionModels.generate_image_from_noise() : Generates a new image from random noise and de-noises it.


"""Getting Started"""

Use the function ImageGenerationDiffusionModels.apply_noise(ImageGenerationDiffusionModels.generate_grid()).
You should be able to see grid.png, the imported image, and noisy_img.png, the image with noise.
