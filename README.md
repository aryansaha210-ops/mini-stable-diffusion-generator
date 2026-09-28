# Mini Stable-Diffusion-Style Text-to-Image Generator

<p align="center">
  <b>An educational implementation of a text-conditioned latent diffusion system built from scratch with PyTorch.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.11-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-1.64-FF4B4B?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/Dataset-MNIST-orange" alt="MNIST">
</p>

---

## Overview

This project is a **from-scratch educational implementation of a Stable-Diffusion-style text-to-image generation pipeline** using PyTorch.

The system demonstrates how several modern generative AI components work together:

- Variational Autoencoder (VAE)
- Latent-space diffusion
- U-Net denoising
- Transformer-based text encoding
- Cross-attention
- Classifier-Free Guidance (CFG)
- Conditional generation
- ControlNet-style conditioning experiments
- LoRA experimentation
- Automated generated-image evaluation
- Streamlit deployment

The project was developed progressively, starting from a basic DDPM and evolving into a text-conditioned latent diffusion pipeline.

> **Note:** This is not a reproduction of the full Stable Diffusion model. It is a small educational system trained on MNIST handwritten digits to demonstrate the underlying concepts.

---

# Demo

The project includes an interactive **Streamlit application** that allows users to enter a text prompt and generate a corresponding handwritten digit.

### Example prompt

```text
a handwritten digit seven


The application performs the following pipeline:

Text Prompt
     │
     ▼
Tokenizer
     │
     ▼
Transformer Text Encoder
     │
     ▼
Text Embeddings
     │
     ▼
Latent Diffusion U-Net
     │
     ▼
Denoised Latent
     │
     ▼
Spatial VAE Decoder
     │
     ▼
Generated 32 × 32 Image


Architecture

                         ┌──────────────────┐
                         │   Text Prompt    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    Tokenizer     │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ Transformer Text Encoder│
                    └────────────┬─────────────┘
                                 │
                                 ▼
                         Text Embeddings
                                 │
                                 │
                                 ▼
┌────────────────┐       ┌─────────────────────┐
│ Random Latent  │──────►│  Latent Diffusion   │
│    Noise       │       │       U-Net         │
└────────────────┘       │                     │
                         │ • Time Embeddings   │
                         │ • Residual Blocks   │
                         │ • Cross-Attention   │
                         │ • Skip Connections  │
                         │ • CFG Conditioning  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                           Denoised Latent
                                    │
                                    ▼
                         ┌──────────────────┐
                         │    Spatial VAE   │
                         │     Decoder      │
                         └────────┬─────────┘
                                  │
                                  ▼
                           32 × 32 Image



Core Components
1. Variational Autoencoder

The VAE compresses the input image into a spatial latent representation.

Encoder
32 × 32 × 1
     │
     ▼
Conv 1 → 32
     │
     ▼
Conv 32 → 64
     │
     ▼
Conv 64 → 128
     │
     ├──────────────► μ
     │
     └──────────────► log variance
                          │
                          ▼
                       4 × 8 × 8
Decoder
4 × 8 × 8
    │
    ▼
Conv 4 → 128
    │
    ▼
Transposed Conv
    │
    ▼
Transposed Conv
    │
    ▼
Conv → 1
    │
    ▼
32 × 32 × 1

The trained VAE achieved an average reconstruction MSE of approximately:

0.00181
2. Latent Diffusion

Instead of performing the complete diffusion process directly on pixels, the project performs diffusion in the VAE latent space.

Image
  │
  ▼
VAE Encoder
  │
  ▼
Latent Representation
  │
  ▼
Add Noise
  │
  ▼
Latent Diffusion
  │
  ▼
Denoised Latent
  │
  ▼
VAE Decoder
  │
  ▼
Image
Diffusion configuration
Parameter	Value
Timesteps	300
β start	0.0001
β end	0.02
Latent channels	4
Latent spatial size	8 × 8
3. Transformer Text Encoder

Text prompts are converted into token IDs and processed by a Transformer encoder.

Configuration
Component	Value
Embedding dimension	128
Attention heads	4
Transformer layers	2
Feed-forward dimension	256
Maximum tokens	16

The vocabulary contains prompt tokens such as:

a
handwritten
digit
zero
one
two
three
four
five
six
seven
eight
nine

This vocabulary is intentionally small because the model is trained on MNIST.

4. Cross-Attention

Cross-attention connects the text representation with the latent image representation.

Latent Features ──────► Query
Text Embeddings ──────► Key
Text Embeddings ──────► Value

              │
              ▼
       Cross-Attention
              │
              ▼
    Text-Conditioned Features

This allows the U-Net to incorporate information from the text prompt during denoising.

5. Conditional U-Net

The diffusion model uses a compact U-Net architecture.

Main components include:

Input convolution
Residual blocks
Time embedding
Downsampling
Middle blocks
Cross-attention
Upsampling
Skip connections
Output convolution

The final U-Net contains approximately:

2.06 million parameters
6. Classifier-Free Guidance

Classifier-Free Guidance (CFG) is used to control the influence of the text prompt during generation.

During training, the model learns both:

Conditional prediction

and:

Unconditional prediction

During sampling, these predictions are combined to strengthen conditioning.

The Streamlit interface allows CFG strength to be adjusted during generation.

Development Progression

The project was developed in multiple stages:

Basic DDPM
    │
    ▼
Conditional Diffusion
    │
    ▼
VAE
    │
    ▼
Latent Diffusion
    │
    ▼
Text Conditioning
    │
    ▼
Spatial Latent U-Net
    │
    ▼
Real Latent Diffusion
    │
    ▼
Classifier-Free Guidance
    │
    ▼
ControlNet-Style Experiment
    │
    ▼
Evaluation
    │
    ▼
Streamlit Application

This progressive approach made it possible to study each major component independently before combining them.

Dataset

The project uses the MNIST handwritten digit dataset.

Dataset characteristics
Property	Value
Classes	10
Classes	0–9
Original resolution	28 × 28
Channels	1
Image type	Grayscale
Model resolution	32 × 32

Example prompts:

a handwritten digit zero
a handwritten digit one
a handwritten digit seven
a handwritten digit nine
Training Configuration
Parameter	Value
Batch size	128
Learning rate	1 × 10⁻⁴
Diffusion timesteps	300
Latent channels	4
Text dimension	128
Maximum tokens	16
CFG dropout	0.15
Image resolution	32 × 32
Safe Fine-Tuning Experiment

A controlled fine-tuning experiment was performed using the best early checkpoint.

Configuration
Starting checkpoint:
cfg_text_real_latent_BEST.pth

Learning rate:
1 × 10⁻⁵

Epochs:
3

Text encoder:
Frozen

VAE:
Frozen
Validation results
Stage	Validation MSE
Initial checkpoint	0.140280
Epoch 1	0.136287
Epoch 2	0.136134
Epoch 3	0.135713

The best validation checkpoint was retained as the final model.

Evaluation

Generated images were evaluated using an independently trained CNN classifier.

The classifier achieved:

MNIST test accuracy: 99.11%

For the generation evaluation:

10 digits × 10 generated samples
= 100 generated images
Overall results
Metric	Result
Images evaluated	100
Correct predictions	68
Incorrect predictions	32
Generation accuracy	68.00%
Macro Precision	71.41%
Macro Recall	68.00%
Macro F1	66.64%
Average CNN confidence	85.61%
CNN test accuracy	99.11%
Per-digit accuracy
Digit	Accuracy
0	60%
1	90%
2	30%
3	90%
4	80%
5	60%
6	100%
7	80%
8	30%
9	60%

The results demonstrate that the model learned recognizable digit structures, while also showing substantial variation between digit classes.

Evaluation Methodology

The diffusion model is evaluated using a separate CNN rather than using its own predictions.

                    Text Prompt
                         │
                         ▼
                 Diffusion Model
                         │
                         ▼
                  Generated Image
                         │
                         ▼
                  Independent CNN
                         │
                         ▼
                  Predicted Digit
                         │
                         ▼
                 Compare with Target

This provides an independent quantitative measurement of generation quality.

Streamlit Application

The project includes an interactive Streamlit interface.

Features
Text prompt input
CFG scale control
CUDA/CPU detection
GPU information
Latent diffusion generation
Generated image visualization
Model architecture information
Supported digit prompts
Run the application
streamlit run app\streamlit_app.py
Project Structure
mini-stable-diffusion-generator/
│
├── app/
│   ├── app.py
│   └── streamlit_app.py
│
├── src/
│   ├── diffusion implementations
│   ├── VAE implementations
│   ├── latent diffusion
│   ├── text conditioning
│   ├── CFG experiments
│   ├── ControlNet-style experiments
│   ├── evaluation scripts
│   └── utility scripts
│
├── requirements.txt
├── README.md
└── .gitignore

Large model checkpoints, generated images, temporary files, virtual environments, and archived experiments are intentionally excluded from Git.

Installation
1. Clone the repository
git clone https://github.com/aryansaha210-ops/mini-stable-diffusion-generator.git
cd mini-stable-diffusion-generator
2. Create a virtual environment
Windows
python -m venv venv

Activate it:

.\venv\Scripts\Activate.ps1
3. Install dependencies
pip install -r requirements.txt

PyTorch installation may need to be adjusted according to the available operating system, GPU, and CUDA configuration.

4. Launch the application
streamlit run app\streamlit_app.py
Hardware

Development and training were performed using:

GPU:
NVIDIA GeForce RTX 3050 Laptop GPU

VRAM:
6 GB

Framework:
PyTorch

CUDA:
CUDA-enabled PyTorch

The project can also run on CPU, although training and generation will be considerably slower.

Technologies
Python
PyTorch
TorchVision
Hugging Face Diffusers
Hugging Face Transformers
Accelerate
NumPy
Pandas
Matplotlib
Scikit-learn
Pillow
Streamlit
CUDA
Experiments

The project includes experiments covering:

Diffusion
Forward diffusion
Reverse diffusion
Noise prediction
DDPM sampling
VAE
Encoder
Latent mean and variance
Reparameterization
Decoder
Reconstruction loss
KL divergence
Latent Diffusion
VAE latent encoding
Latent noise injection
Latent U-Net denoising
Latent decoding
Text Conditioning
Deterministic tokenizer
Transformer text encoder
Text embeddings
Cross-attention
Advanced Conditioning
Classifier-Free Guidance
LoRA experimentation
ControlNet-style conditioning
Limitations

This project intentionally operates at a small educational scale.

Limited Dataset

The model is trained on MNIST digits and therefore does not learn a broad visual vocabulary.

It should not be expected to generate arbitrary concepts such as:

cats
cars
landscapes
people
photographs
complex objects
Low Resolution

The system generates:

32 × 32

images.

Small Model

The U-Net contains approximately 2.06 million parameters, which is extremely small compared with production-scale diffusion systems.

Limited Text Encoder

The tokenizer and Transformer are specialized for the small MNIST vocabulary rather than general natural-language understanding.

Not Full Stable Diffusion

The project demonstrates the major concepts behind Stable-Diffusion-style systems but is not a production implementation of Stable Diffusion.

Future Improvements

Potential future work includes:

Training on larger and more diverse datasets
Increasing image resolution
Improving the tokenizer
Using a stronger text encoder
Increasing U-Net capacity
Improving the VAE
Experimenting with improved noise schedules
Implementing DDIM sampling
Improving CFG training
More robust LoRA implementation
More faithful ControlNet implementation
Attention visualization
Experiment tracking
Automated benchmarking
Web deployment
Model optimization for low-VRAM GPUs
Learning Outcomes

This project provided practical experience with:

Diffusion probabilistic models
DDPM
Variational Autoencoders
Latent representations
U-Net architectures
Residual networks
Transformers
Self-attention
Cross-attention
Text conditioning
Classifier-Free Guidance
Conditional generation
Fine-tuning
LoRA concepts
ControlNet concepts
Model evaluation
Confusion matrices
GPU acceleration
CUDA
PyTorch model serialization
Streamlit deployment
Why This Project?

The primary goal was to understand the architecture behind modern generative AI systems rather than treating a pretrained pipeline as a black box.

The project progressively builds the concepts:

Noise
  ↓
Diffusion
  ↓
VAE
  ↓
Latent Diffusion
  ↓
Text Conditioning
  ↓
Cross-Attention
  ↓
Classifier-Free Guidance
  ↓
Evaluation
  ↓
Interactive Generation

This makes the repository useful as both an educational implementation and an experimentation platform for understanding diffusion-based generative models.

Author
Aryan Saha

GitHub:
https://github.com/aryansaha210-ops

Project:
https://github.com/aryansaha210-ops/mini-stable-diffusion-generator

License

No license has currently been specified for this repository.


### Then save and push it

```powershell
cd D:\StableDiffusionProject
notepad .\README.md