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
img = generate_grid()
```
```julia
noisy_img = apply_noise(img)
```
```julia
train_brain()
```
```julia
denoised_img = denoise_image(img[1:32, 1:32])
```
```julia
new_img = generate_image_from_noise()
```
The last two functions create a png with the same name so you can only see one at the time !
