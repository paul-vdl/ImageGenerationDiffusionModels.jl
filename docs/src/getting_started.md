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
