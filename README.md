# Mini Stable-Diffusion-Style Text-to-Image Generator



A complete educational implementation of a Stable-Diffusion-style text-to-image pipeline built from scratch using PyTorch.



The project demonstrates how modern diffusion systems can be constructed step by step, including variational autoencoders, latent diffusion, text conditioning, cross-attention, classifier-free guidance, evaluation, and a Streamlit interface.



> \*\*Important:\*\* This is an educational miniature diffusion system, not the original Stable Diffusion model. It is trained on handwritten digit data and therefore supports a limited vocabulary of digit-related prompts.



\---



\## Features



\- Custom Variational Autoencoder (VAE)

\- Spatial latent representation

\- Latent diffusion model

\- Text conditioning

\- Deterministic tokenizer

\- Transformer-based text encoder

\- Cross-attention

\- Classifier-Free Guidance (CFG)

\- ControlNet-style conditioning experiment

\- Safe fine-tuning

\- Independent CNN evaluation

\- Quantitative evaluation

\- Confusion matrix

\- Per-digit accuracy analysis

\- Training-loss visualization

\- Streamlit web interface

\- GPU acceleration with CUDA

\- Complete PyTorch implementation



\---



\## Project Architecture



The complete pipeline is:



```text

Text Prompt

&#x20;   │

&#x20;   ▼

Tokenizer

&#x20;   │

&#x20;   ▼

Text Encoder

&#x20;   │

&#x20;   ▼

Text Embeddings

&#x20;   │

&#x20;   │

&#x20;   ├──────────────────────┐

&#x20;   │                      │

&#x20;   ▼                      ▼

Noise Latent          Cross-Attention

&#x20;   │                      │

&#x20;   └──────────┬───────────┘

&#x20;              ▼

&#x20;       Latent Diffusion

&#x20;          U-Net

&#x20;              │

&#x20;              ▼

&#x20;      Denoised Latent

&#x20;              │

&#x20;              ▼

&#x20;           VAE

&#x20;          Decoder

&#x20;              │

&#x20;              ▼

&#x20;       Generated Image

