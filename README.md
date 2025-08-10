# 🎨 Text-to-Image Generation with Stable Diffusion 🤖

This project provides a straightforward implementation of a text-to-image generation pipeline using the Hugging Face `diffusers` library. The script leverages a pre-trained Stable Diffusion model (`dreamlike-art/dreamlike-diffusion-1.0`) to create high-quality, artistic images from textual descriptions (prompts). ✨

The included Jupyter/Colab notebook (`Image_Generation_from_Text.ipynb`) demonstrates the entire process, from setting up the environment to generating images with various customizable parameters.

## ✨ Features

  * 🧩 Utilizes the powerful `diffusers` library for easy access to diffusion models.
  * 🖼️ Loads the `dreamlike-art/dreamlike-diffusion-1.0` model from the Hugging Face Hub.
  * 🚀 Optimized for GPU usage with `CUDA` and `float16` precision for faster inference.
  * 🛠️ A helper function `generate_image` to simplify the generation process and display results.
  * 🎛️ Demonstrates how to control key generation parameters:
      * `num_inference_steps`: The number of denoising steps.
      * `width` & `height`: The dimensions of the output image.
      * `num_images_per_prompt`: The number of images to generate for a single prompt.

## ✅ Prerequisites

  * 🐍 Python 3.7+
  * 💻 An NVIDIA GPU with CUDA support is highly recommended for reasonable generation times.
  * 📓 The script is designed to be run in an environment like Google Colab or a local Jupyter Notebook setup.

## 🔧 Installation

1.  **Clone the repository or download the `.ipynb` file.** 📂

2.  **Install the required Python libraries using pip:**

    ```bash
    pip install transformers diffusers accelerate torch
    ```

## 🚀 Usage

1.  **Open the `Image_Generation_from_Text.ipynb` notebook** in Google Colab or a local Jupyter environment.

2.  **Run the cells sequentially.** ▶️ The notebook will handle:

      * Installing dependencies.
      * Loading the pre-trained model onto the GPU.
      * Defining the `generate_image` helper function.

3.  **To generate your own images, modify the `prompt` and `params` variables and run the cell.** ✍️

### 🌟 Example

Here is a complete example of how to generate 3 images with custom dimensions and inference steps:

```python
# 1. 📝 Define your creative prompt
prompt = "A majestic cyberpunk lion with neon circuits, standing on a rainy futuristic street, in the style of Blade Runner, 8k, highly detailed"

# 2. ⚙️ Define the generation parameters
# Note: The typo 'num_inference_step' from the original notebook is corrected to 'num_inference_steps' here.
params = {
    'num_inference_steps': 150,
    'width': 768,
    'height': 512,
    'num_images_per_prompt': 3
}

# 3. ✨ Generate and display the images
# The 'pipe' object should already be loaded and available from previous cells.
generate_image(pipe, prompt, params)
```

## 🧑‍🏫 Code Explanation

### 1\. Model Initialization

```python
from diffusers import StableDiffusionPipeline
import torch

# Define the model ID from Hugging Face Hub
model_id = "dreamlike-art/dreamlike-diffusion-1.0"

# Load the pretrained pipeline
# torch.float16 uses less VRAM and is faster on modern GPUs ⚡
# use_safetensors=True is for safer model loading 🔒
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True
)

# Move the pipeline to the GPU for hardware acceleration
pipe = pipe.to("cuda")
```

This section loads the `dreamlike-diffusion-1.0` model and moves it to the GPU (`cuda`) for efficient processing.

### 2\. The `generate_image` Function

```python
import matplotlib.pyplot as plt

def generate_image(pipe, prompt, params):
  # Generate image(s) by passing the prompt and unpacking the params dictionary
  images = pipe(prompt, **params).images

  num_images = len(images)

  # Use matplotlib to display the generated images in a grid if there are multiple
  if(num_images > 1):
    fig, ax = plt.subplots(nrows=1, ncols=num_images, figsize=(15, 5))
    for i in range(num_images):
      ax[i].imshow(images[i])
      ax[i].axis("off")
  else:
    # Display a single image
    plt.figure()
    plt.imshow(images[0])
    plt.axis('off')

  plt.tight_layout()
```

This helper function takes the model pipeline, a prompt, and a dictionary of parameters. It calls the pipeline, then uses `matplotlib` to plot and display the resulting image or images.

## 🎛️ Parameter Reference

You can customize the image generation process by passing different arguments in the `params` dictionary.

  * `num_inference_steps` 🔢 (integer, default: 50)

      * The number of denoising steps. Higher values can lead to more detailed and higher-quality images but take longer to generate. Values between 50 and 150 are common.

  * `width`, `height` 📏 (integer, default: 512)

      * The dimensions of the output image in pixels. For best results, use values that are multiples of 8.

  * `num_images_per_prompt` 🖼️ (integer, default: 1)

      * The number of distinct images to generate from the same prompt in a single run.

  * `guidance_scale` 🧭 (float, default: 7.5)

      * A value that controls how much the generation process adheres to the prompt. Higher values mean the model follows the prompt more strictly, while lower values allow for more creative freedom.
## Some Images generated
<img width="640" height="640" alt="download" src="https://github.com/user-attachments/assets/dff04934-6d50-4b91-aabd-f6438f56a048" />

-----
