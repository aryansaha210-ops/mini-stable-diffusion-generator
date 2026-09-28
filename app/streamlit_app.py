import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TIMESTEPS = 300
LATENT_CHANNELS = 4
TEXT_DIM = 128
VOCAB_SIZE = 10000
MAX_TOKENS = 16

CHECKPOINT_PATH = os.path.join(
    "checkpoints",
    "cfg_text_real_latent_FINAL.pth"
)

VAE_CHECKPOINT_PATH = os.path.join(
    "checkpoints",
    "spatial_vae_epoch_10.pth"
)


# ============================================================
# VOCABULARY / TOKENIZER
# ============================================================

VOCAB = {
    "a": 10,
    "handwritten": 11,
    "digit": 12,
    "zero": 13,
    "one": 14,
    "two": 15,
    "three": 16,
    "four": 17,
    "five": 18,
    "six": 19,
    "seven": 20,
    "eight": 21,
    "nine": 22,
}

PAD = 0
UNK = 1
START = 2
END = 3


def tokenize(text):

    words = text.lower().split()

    tokens = [START]

    for word in words:
        tokens.append(
            VOCAB.get(word, UNK)
        )

    tokens.append(END)

    tokens = tokens[:MAX_TOKENS]

    while len(tokens) < MAX_TOKENS:
        tokens.append(PAD)

    return torch.tensor(
        tokens,
        dtype=torch.long
    )


# ============================================================
# TIME EMBEDDING
# ============================================================

class TimeEmbedding(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )

    def forward(self, t):

        t = t.float().unsqueeze(1)

        t = t / TIMESTEPS

        return self.network(t)


# ============================================================
# TEXT ENCODER
#
# IMPORTANT:
# The trained checkpoint uses:
# encoder.layers.*
#
# Therefore this module must use self.encoder,
# not self.transformer.
# ============================================================

class TextEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            TEXT_DIM
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TEXT_DIM,
            nhead=4,
            dim_feedforward=256,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

    def forward(self, tokens):

        x = self.embedding(tokens)

        x = self.encoder(x)

        return x


# ============================================================
# CROSS ATTENTION
# ============================================================

class CrossAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.query = nn.Linear(
            128,
            128
        )

        self.key = nn.Linear(
            TEXT_DIM,
            128
        )

        self.value = nn.Linear(
            TEXT_DIM,
            128
        )

        self.output = nn.Linear(
            128,
            128
        )

        self.num_heads = 4

        self.head_dim = 32

    def forward(
        self,
        x,
        text
    ):

        B, C, H, W = x.shape

        x_flat = x.permute(
            0,
            2,
            3,
            1
        ).reshape(
            B,
            H * W,
            C
        )

        Q = self.query(x_flat)

        K = self.key(text)

        V = self.value(text)

        Q = Q.view(
            B,
            H * W,
            self.num_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )

        K = K.view(
            B,
            MAX_TOKENS,
            self.num_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )

        V = V.view(
            B,
            MAX_TOKENS,
            self.num_heads,
            self.head_dim
        ).transpose(
            1,
            2
        )

        scores = torch.matmul(
            Q,
            K.transpose(
                -2,
                -1
            )
        )

        scores = scores / (
            self.head_dim ** 0.5
        )

        attention = torch.softmax(
            scores,
            dim=-1
        )

        out = torch.matmul(
            attention,
            V
        )

        out = out.transpose(
            1,
            2
        ).contiguous()

        out = out.view(
            B,
            H * W,
            128
        )

        out = self.output(out)

        out = out.reshape(
            B,
            H,
            W,
            128
        )

        out = out.permute(
            0,
            3,
            1,
            2
        )

        # IMPORTANT:
        # CrossAttention already includes the residual.
        return x + out


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1
        )

        self.time_projection = nn.Linear(
            128,
            out_channels
        )

        self.text_projection = nn.Linear(
            128,
            out_channels
        )

        if in_channels != out_channels:

            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                1
            )

        else:

            self.skip = nn.Identity()

    def forward(
        self,
        x,
        time_emb,
        text_emb
    ):

        h = self.conv1(x)

        h = F.silu(h)

        time_condition = self.time_projection(
            time_emb
        ).unsqueeze(
            -1
        ).unsqueeze(
            -1
        )

        text_condition = self.text_projection(
            text_emb
        ).unsqueeze(
            -1
        ).unsqueeze(
            -1
        )

        h = (
            h
            + time_condition
            + text_condition
        )

        h = self.conv2(h)

        h = F.silu(h)

        return h + self.skip(x)


# ============================================================
# TEXT LATENT DIFFUSION U-NET
# ============================================================

class TextLatentDiffusionUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.input = nn.Conv2d(
            4,
            64,
            3,
            padding=1
        )

        self.res1 = ResBlock(
            64,
            128
        )

        self.down = nn.Conv2d(
            128,
            128,
            3,
            stride=2,
            padding=1
        )

        self.middle1 = ResBlock(
            128,
            128
        )

        self.cross_attention = CrossAttention()

        self.middle2 = ResBlock(
            128,
            128
        )

        self.up = nn.ConvTranspose2d(
            128,
            128,
            4,
            stride=2,
            padding=1
        )

        self.res2 = ResBlock(
            256,
            128
        )

        self.res3 = ResBlock(
            128,
            64
        )

        self.output = nn.Conv2d(
            64,
            4,
            3,
            padding=1
        )

        self.time_embedding = TimeEmbedding()

    def forward(
        self,
        x,
        t,
        text
    ):

        time_emb = self.time_embedding(t)

        text_mean = text.mean(
            dim=1
        )

        x0 = self.input(x)

        x1 = self.res1(
            x0,
            time_emb,
            text_mean
        )

        skip = x1

        x2 = self.down(x1)

        x2 = self.middle1(
            x2,
            time_emb,
            text_mean
        )

        x2 = self.cross_attention(
            x2,
            text
        )

        x2 = self.middle2(
            x2,
            time_emb,
            text_mean
        )

        x2 = self.up(x2)

        x2 = torch.cat(
            [
                x2,
                skip
            ],
            dim=1
        )

        x2 = self.res2(
            x2,
            time_emb,
            text_mean
        )

        x2 = self.res3(
            x2,
            time_emb,
            text_mean
        )

        return self.output(x2)


# ============================================================
# SPATIAL VAE
# ============================================================

class SpatialVAE(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                128,
                3,
                1,
                1
            ),

            nn.ReLU()
        )

        self.mu = nn.Conv2d(
            128,
            4,
            kernel_size=3,
            padding=1
        )

        self.logvar = nn.Conv2d(
            128,
            4,
            kernel_size=3,
            padding=1
        )

        self.decoder = nn.Sequential(

            nn.Conv2d(
                4,
                128,
                3,
                1,
                1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                128,
                64,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                1,
                3,
                1,
                1
            ),

            nn.Tanh()
        )

    def encode(self, x):

        h = self.encoder(x)

        return self.mu(h)

    def decode(self, z):

        return self.decoder(z)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    # --------------------------------------------------------
    # LOAD VAE
    # --------------------------------------------------------

    vae = SpatialVAE().to(
        DEVICE
    )

    vae_checkpoint = torch.load(
        VAE_CHECKPOINT_PATH,
        map_location=DEVICE
    )

    if (
        isinstance(
            vae_checkpoint,
            dict
        )
        and "model" in vae_checkpoint
    ):

        vae.load_state_dict(
            vae_checkpoint["model"]
        )

    else:

        vae.load_state_dict(
            vae_checkpoint
        )

    vae.eval()

    # --------------------------------------------------------
    # LOAD DIFFUSION MODEL
    # --------------------------------------------------------

    model = TextLatentDiffusionUNet().to(
        DEVICE
    )

    # --------------------------------------------------------
    # LOAD TEXT ENCODER
    # --------------------------------------------------------

    text_encoder = TextEncoder().to(
        DEVICE
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    text_encoder.load_state_dict(
        checkpoint["text_encoder"]
    )

    model.eval()

    text_encoder.eval()

    return (
        vae,
        model,
        text_encoder
    )


# ============================================================
# DIFFUSION SCHEDULE
# ============================================================

betas = torch.linspace(
    1e-4,
    0.02,
    TIMESTEPS,
    device=DEVICE
)

alphas = 1.0 - betas

alpha_bars = torch.cumprod(
    alphas,
    dim=0
)


# ============================================================
# GENERATE IMAGE
# ============================================================

@torch.no_grad()
def generate_image(
    prompt,
    guidance_scale
):

    vae, model, text_encoder = load_models()

    # --------------------------------------------------------
    # CONDITIONAL PROMPT
    # --------------------------------------------------------

    tokens = tokenize(
        prompt
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    # --------------------------------------------------------
    # EMPTY PROMPT FOR CFG
    # --------------------------------------------------------

    empty_tokens = tokenize(
        ""
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    conditional_text = text_encoder(
        tokens
    )

    unconditional_text = text_encoder(
        empty_tokens
    )

    # --------------------------------------------------------
    # INITIAL RANDOM LATENT
    # --------------------------------------------------------

    latent = torch.randn(
        1,
        LATENT_CHANNELS,
        8,
        8,
        device=DEVICE
    )

    progress = st.progress(
        0
    )

    # --------------------------------------------------------
    # REVERSE DIFFUSION
    # --------------------------------------------------------

    for step, t_value in enumerate(
        reversed(
            range(TIMESTEPS)
        )
    ):

        t = torch.tensor(
            [t_value],
            device=DEVICE
        )

        # Conditional prediction
        conditional_noise = model(
            latent,
            t,
            conditional_text
        )

        # Unconditional prediction
        unconditional_noise = model(
            latent,
            t,
            unconditional_text
        )

        # Classifier-Free Guidance
        guided_noise = (
            unconditional_noise
            + guidance_scale
            * (
                conditional_noise
                - unconditional_noise
            )
        )

        beta_t = betas[
            t_value
        ]

        alpha_t = alphas[
            t_value
        ]

        alpha_bar_t = alpha_bars[
            t_value
        ]

        if t_value > 0:

            alpha_bar_prev = alpha_bars[
                t_value - 1
            ]

        else:

            alpha_bar_prev = torch.tensor(
                1.0,
                device=DEVICE
            )

        # ----------------------------------------------------
        # DDPM POSTERIOR
        # ----------------------------------------------------

        posterior_variance = (
            beta_t
            * (
                1.0
                - alpha_bar_prev
            )
            / (
                1.0
                - alpha_bar_t
            )
        )

        posterior_mean = (
            1.0
            / torch.sqrt(
                alpha_t
            )
        ) * (
            latent
            - (
                beta_t
                / torch.sqrt(
                    1.0
                    - alpha_bar_t
                )
            )
            * guided_noise
        )

        if t_value > 0:

            noise = torch.randn_like(
                latent
            )

            latent = (
                posterior_mean
                + torch.sqrt(
                    posterior_variance
                )
                * noise
            )

        else:

            latent = posterior_mean

        progress.progress(
            (step + 1)
            / TIMESTEPS
        )

    # --------------------------------------------------------
    # DECODE LATENT
    # --------------------------------------------------------

    image = vae.decode(
        latent
    )

    image = image.clamp(
        -1,
        1
    )

    image = (
        image + 1
    ) / 2

    image = image.squeeze(
        0
    ).squeeze(
        0
    )

    image = (
        image.cpu()
        .numpy()
        * 255
    ).astype(
        "uint8"
    )

    return Image.fromarray(
        image,
        mode="L"
    )


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Mini Stable Diffusion",
    page_icon="🎨",
    layout="centered"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🎨 Mini Stable-Diffusion-Style Generator"
)

st.write(
    "Educational text-conditioned latent diffusion "
    "system trained on MNIST handwritten digits."
)

st.info(
    "This model was trained only on handwritten digits. "
    "Use prompts such as "
    "'a handwritten digit seven'."
)

st.divider()


# ============================================================
# GENERATOR
# ============================================================

st.subheader(
    "Generate an Image"
)

prompt = st.text_input(
    "Prompt",
    value="a handwritten digit seven"
)

guidance_scale = st.slider(
    "CFG Guidance Scale",
    min_value=1.0,
    max_value=5.0,
    value=2.0,
    step=0.5
)

st.caption(
    "Recommended CFG scale: 2.0"
)

generate_button = st.button(
    "🚀 Generate",
    type="primary"
)


# ============================================================
# GENERATION ACTION
# ============================================================

if generate_button:

    if prompt.strip() == "":

        st.warning(
            "Please enter a prompt."
        )

    else:

        try:

            with st.spinner(
                "Loading model and generating..."
            ):

                image = generate_image(
                    prompt,
                    guidance_scale
                )

            st.success(
                "Generation complete!"
            )

            st.image(
                image,
                caption=prompt,
                width=256
            )

            # ------------------------------------------------
            # GENERATION INFORMATION
            # ------------------------------------------------

            st.subheader(
                "Generation Information"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Diffusion Steps",
                    TIMESTEPS
                )

            with col2:

                st.metric(
                    "Latent Size",
                    "4 × 8 × 8"
                )

            st.write(
                "**Checkpoint:** "
                "`cfg_text_real_latent_FINAL.pth`"
            )

            st.write(
                "**VAE:** "
                "`spatial_vae_epoch_10.pth`"
            )

            st.write(
                "**Device:** "
                + DEVICE
            )

            if DEVICE == "cuda":

                st.write(
                    "**GPU:** "
                    + torch.cuda.get_device_name(
                        0
                    )
                )

        except Exception as e:

            st.error(
                "Generation failed."
            )

            st.exception(e)


# ============================================================
# SUPPORTED DIGITS
# ============================================================

st.divider()

st.subheader(
    "Supported Digits"
)

st.write(
    "The model was trained on handwritten MNIST digits:"
)

st.write(
    "0 • 1 • 2 • 3 • 4 • 5 • 6 • 7 • 8 • 9"
)

st.caption(
    "Example prompts:"
)

st.code(
    "a handwritten digit zero\n"
    "a handwritten digit three\n"
    "a handwritten digit seven\n"
    "a handwritten digit nine"
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader(
    "Model Architecture"
)

st.write(
    """
**Text Encoder**
- Token embedding
- 2-layer Transformer encoder
- 128-dimensional text representation

**Latent Diffusion**
- 4-channel spatial latent
- 8 × 8 latent resolution
- Conditional U-Net
- Cross-attention
- Classifier-Free Guidance
- 300 diffusion timesteps

**Spatial VAE**
- 32 × 32 image input
- 4-channel latent representation
- 8 × 8 latent spatial resolution
- Decoder reconstructs 32 × 32 image
"""
)

st.caption(
    "Educational miniature Stable-Diffusion-style "
    "architecture — not the original Stable Diffusion model."
)