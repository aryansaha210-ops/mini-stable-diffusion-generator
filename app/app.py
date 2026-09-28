import os
import math

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import datasets, transforms


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TIMESTEPS = 300

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

VAE_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "spatial_vae_epoch_10.pth"
)

CFG_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "cfg_text_real_latent_epoch_5.pth"
)

CONTROLNET_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "controlnet_epoch_5.pth"
)


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Stable Diffusion Style AI",
    page_icon="🎨",
    layout="wide"
)

st.title(
    "🎨 Stable Diffusion-Style Generator"
)

st.caption(
    "Educational text-conditioned latent diffusion "
    "with Classifier-Free Guidance and ControlNet-style conditioning."
)


# ============================================================
# DEVICE INFORMATION
# ============================================================

if DEVICE == "cuda":

    st.success(
        "🚀 NVIDIA CUDA GPU detected: "
        + torch.cuda.get_device_name(0)
    )

else:

    st.warning(
        "⚠️ CUDA not detected. Running on CPU."
    )


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
            3,
            padding=1
        )

        self.logvar = nn.Conv2d(
            128,
            4,
            3,
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

        mu = self.mu(h)

        logvar = self.logvar(h)

        return mu, logvar

    def decode(self, z):

        return self.decoder(z)


# ============================================================
# DETERMINISTIC TOKENIZER
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
    "nine": 22
}

PAD_TOKEN = 0
UNK_TOKEN = 1
START_TOKEN = 2
END_TOKEN = 3

MAX_TOKENS = 16

VOCAB_SIZE = 10000


def tokenize(text):

    words = (
        text
        .lower()
        .strip()
        .split()
    )

    tokens = [
        START_TOKEN
    ]

    for word in words:

        tokens.append(
            VOCAB.get(
                word,
                UNK_TOKEN
            )
        )

    tokens.append(
        END_TOKEN
    )

    tokens = tokens[:MAX_TOKENS]

    while len(tokens) < MAX_TOKENS:

        tokens.append(
            PAD_TOKEN
        )

    return torch.tensor(
        tokens,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)


# ============================================================
# TEXT ENCODER
# ============================================================

class TextEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            128
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                batch_first=True
            )
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

    def forward(self, tokens):

        x = self.embedding(
            tokens
        )

        x = self.encoder(
            x
        )

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
            128,
            128
        )

        self.value = nn.Linear(
            128,
            128
        )

        self.output = nn.Linear(
            128,
            128
        )

        self.heads = 4

        self.head_dim = 32

    def forward(
        self,
        x,
        text
    ):

        B, C, H, W = x.shape

        x_flat = (
            x
            .flatten(2)
            .transpose(1, 2)
        )

        Q = self.query(
            x_flat
        )

        K = self.key(
            text
        )

        V = self.value(
            text
        )

        Q = Q.view(
            B,
            -1,
            self.heads,
            self.head_dim
        ).transpose(
            1,
            2
        )

        K = K.view(
            B,
            -1,
            self.heads,
            self.head_dim
        ).transpose(
            1,
            2
        )

        V = V.view(
            B,
            -1,
            self.heads,
            self.head_dim
        ).transpose(
            1,
            2
        )

        attention = torch.matmul(
            Q,
            K.transpose(
                -2,
                -1
            )
        )

        attention = (
            attention
            / math.sqrt(
                self.head_dim
            )
        )

        attention = torch.softmax(
            attention,
            dim=-1
        )

        result = torch.matmul(
            attention,
            V
        )

        result = (
            result
            .transpose(1, 2)
            .contiguous()
        )

        result = result.view(
            B,
            -1,
            128
        )

        result = self.output(
            result
        )

        result = (
            result
            .transpose(1, 2)
            .reshape(
                B,
                128,
                H,
                W
            )
        )

        return result


# ============================================================
# TIME EMBEDDING
# ============================================================

class TimeEmbedding(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                1,
                128
            ),

            nn.SiLU(),

            nn.Linear(
                128,
                128
            )
        )

    def forward(self, t):

        t = t.float().view(
            -1,
            1
        )

        return self.network(
            t
        )


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
        time_embedding,
        text_embedding
    ):

        h = self.conv1(
            x
        )

        h = (
            h
            + self.time_projection(
                time_embedding
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        h = (
            h
            + self.text_projection(
                text_embedding
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        h = F.silu(
            h
        )

        h = self.conv2(
            h
        )

        h = F.silu(
            h
        )

        return (
            h
            + self.skip(x)
        )


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
        text,
        control=None
    ):

        time_embedding = (
            self.time_embedding(t)
        )

        text_mean = text.mean(
            dim=1
        )

        x0 = self.input(
            x
        )

        x1 = self.res1(
            x0,
            time_embedding,
            text_mean
        )

        x2 = self.down(
            x1
        )

        x3 = self.middle1(
            x2,
            time_embedding,
            text_mean
        )

        if control is not None:

            x3 = x3 + control

        attention = self.cross_attention(
            x3,
            text
        )

        x3 = x3 + attention

        x3 = self.middle2(
            x3,
            time_embedding,
            text_mean
        )

        x4 = self.up(
            x3
        )

        x4 = torch.cat(
            [
                x4,
                x1
            ],
            dim=1
        )

        x4 = self.res2(
            x4,
            time_embedding,
            text_mean
        )

        x4 = self.res3(
            x4,
            time_embedding,
            text_mean
        )

        return self.output(
            x4
        )


# ============================================================
# CONTROLNET-STYLE CONDITIONING
# ============================================================

class ControlNetCondition(nn.Module):

    def __init__(self):

        super().__init__()

        self.condition_encoder = nn.Sequential(

            nn.Conv2d(
                4,
                64,
                3,
                padding=1
            ),

            nn.SiLU(),

            nn.Conv2d(
                64,
                128,
                3,
                padding=1
            ),

            nn.SiLU(),

            nn.Conv2d(
                128,
                128,
                3,
                stride=2,
                padding=1
            ),

            nn.SiLU()
        )

        self.zero_conv = nn.Conv2d(
            128,
            128,
            1
        )

    def forward(
        self,
        condition
    ):

        x = self.condition_encoder(
            condition
        )

        x = self.zero_conv(
            x
        )

        return x


# ============================================================
# LOAD ALL MODELS
# ============================================================

@st.cache_resource
def load_models():

    # --------------------------------------------------------
    # VAE
    # --------------------------------------------------------

    vae = SpatialVAE().to(
        DEVICE
    )

    vae_checkpoint = torch.load(
        VAE_CHECKPOINT,
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
    # CFG MODEL
    # --------------------------------------------------------

    cfg_checkpoint = torch.load(
        CFG_CHECKPOINT,
        map_location=DEVICE
    )

    model = TextLatentDiffusionUNet().to(
        DEVICE
    )

    text_encoder = TextEncoder().to(
        DEVICE
    )

    model.load_state_dict(
        cfg_checkpoint["model"]
    )

    text_encoder.load_state_dict(
        cfg_checkpoint["text_encoder"]
    )

    model.eval()

    text_encoder.eval()


    # --------------------------------------------------------
    # CONTROLNET
    # --------------------------------------------------------

    controlnet = ControlNetCondition().to(
        DEVICE
    )

    controlnet_checkpoint = torch.load(
        CONTROLNET_CHECKPOINT,
        map_location=DEVICE
    )

    if (
        isinstance(
            controlnet_checkpoint,
            dict
        )
        and "model" in controlnet_checkpoint
    ):

        controlnet.load_state_dict(
            controlnet_checkpoint["model"]
        )

    else:

        controlnet.load_state_dict(
            controlnet_checkpoint
        )

    controlnet.eval()


    return (
        vae,
        model,
        text_encoder,
        controlnet
    )


# ============================================================
# IMAGE → LATENT
# ============================================================

@torch.no_grad()
def image_to_latent(
    vae,
    image
):

    if isinstance(
        image,
        torch.Tensor
    ):

        image_tensor = image.to(
            DEVICE
        )

        if image_tensor.dim() == 3:

            image_tensor = (
                image_tensor
                .unsqueeze(0)
            )

    else:

        image = image.convert(
            "L"
        )

        image = image.resize(
            (32, 32)
        )

        transform = transforms.ToTensor()

        image_tensor = transform(
            image
        ).unsqueeze(0)

        image_tensor = (
            image_tensor * 2.0
            - 1.0
        )

        image_tensor = image_tensor.to(
            DEVICE
        )

    mu, _ = vae.encode(
        image_tensor
    )

    return mu


# ============================================================
# PROPER DDPM REVERSE DIFFUSION
# ============================================================

@torch.no_grad()
def generate_image(
    prompt,
    guidance_scale,
    control_latent,
    progress_bar
):

    (
        vae,
        model,
        text_encoder,
        controlnet
    ) = load_models()


    # ========================================================
    # TEXT CONDITION
    # ========================================================

    tokens = tokenize(
        prompt
    )

    text_features = text_encoder(
        tokens
    )


    # ========================================================
    # UNCONDITIONAL CONDITION
    # ========================================================

    unconditional_tokens = tokenize(
        ""
    )

    unconditional_features = text_encoder(
        unconditional_tokens
    )


    # ========================================================
    # CONTROLNET CONDITION
    # ========================================================

    control_features = None

    if control_latent is not None:

        control_features = controlnet(
            control_latent
        )


    # ========================================================
    # DDPM BETA SCHEDULE
    # ========================================================

    betas = torch.linspace(
        1e-4,
        0.02,
        TIMESTEPS,
        device=DEVICE
    )

    alphas = (
        1.0 - betas
    )

    alpha_bars = torch.cumprod(
        alphas,
        dim=0
    )


    # ========================================================
    # PREVIOUS ALPHA BAR
    # ========================================================

    alpha_bars_prev = torch.cat(
        [
            torch.ones(
                1,
                device=DEVICE
            ),

            alpha_bars[:-1]
        ]
    )


    # ========================================================
    # DDPM POSTERIOR VARIANCE
    # ========================================================

    posterior_variance = (

        betas

        *

        (
            1.0
            -
            alpha_bars_prev
        )

        /

        (
            1.0
            -
            alpha_bars
        )
    )


    # ========================================================
    # INITIAL RANDOM LATENT
    # ========================================================

    latent = torch.randn(
        1,
        4,
        8,
        8,
        device=DEVICE
    )


    # ========================================================
    # REVERSE DIFFUSION LOOP
    # ========================================================

    for step_index, timestep in enumerate(
        reversed(
            range(
                TIMESTEPS
            )
        )
    ):

        t = torch.tensor(
            [timestep],
            device=DEVICE,
            dtype=torch.long
        )


        # ====================================================
        # CONDITIONAL NOISE
        # ====================================================

        conditional_noise = model(
            latent,
            t,
            text_features,
            control=control_features
        )


        # ====================================================
        # UNCONDITIONAL NOISE
        # ====================================================

        unconditional_noise = model(
            latent,
            t,
            unconditional_features,
            control=control_features
        )


        # ====================================================
        # CLASSIFIER-FREE GUIDANCE
        # ====================================================

        guided_noise = (

            unconditional_noise

            +

            guidance_scale

            *

            (
                conditional_noise
                -
                unconditional_noise
            )
        )


        # ====================================================
        # CURRENT DDPM VALUES
        # ====================================================

        beta_t = betas[
            timestep
        ]

        alpha_t = alphas[
            timestep
        ]

        alpha_bar_t = alpha_bars[
            timestep
        ]

        alpha_bar_prev_t = alpha_bars_prev[
            timestep
        ]


        # ====================================================
        # PREDICT CLEAN LATENT x0
        # ====================================================

        predicted_x0 = (

            latent

            -

            torch.sqrt(
                1.0
                -
                alpha_bar_t
            )

            *

            guided_noise

        ) / torch.sqrt(
            alpha_bar_t
        )


        # ====================================================
        # CLIP CLEAN LATENT RANGE
        # ====================================================

        predicted_x0 = predicted_x0.clamp(
            -1.0,
            1.0
        )


        # ====================================================
        # POSTERIOR MEAN
        # ====================================================

        coefficient_x0 = (

            torch.sqrt(
                alpha_bar_prev_t
            )

            *

            beta_t

            /

            (
                1.0
                -
                alpha_bar_t
            )
        )


        coefficient_xt = (

            torch.sqrt(
                alpha_t
            )

            *

            (
                1.0
                -
                alpha_bar_prev_t
            )

            /

            (
                1.0
                -
                alpha_bar_t
            )
        )


        posterior_mean = (

            coefficient_x0
            *
            predicted_x0

            +

            coefficient_xt
            *
            latent
        )


        # ====================================================
        # SAMPLE PREVIOUS LATENT
        # ====================================================

        if timestep > 0:

            noise = torch.randn_like(
                latent
            )

            latent = (

                posterior_mean

                +

                torch.sqrt(
                    posterior_variance[
                        timestep
                    ]
                )

                *

                noise
            )

        else:

            latent = posterior_mean


        # ====================================================
        # UPDATE PROGRESS BAR
        # ====================================================

        progress = int(

            (
                step_index
                + 1
            )

            /

            TIMESTEPS

            *

            100
        )

        progress_bar.progress(
            progress
        )


    # ========================================================
    # DECODE FINAL LATENT
    # ========================================================

    image = vae.decode(
        latent
    )

    image = image.clamp(
        -1.0,
        1.0
    )

    image = (
        image + 1.0
    ) / 2.0

    return image


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Generation Settings"
)

guidance_scale = st.sidebar.slider(
    "CFG Scale",
    min_value=1.0,
    max_value=10.0,
    value=5.0,
    step=0.5
)

st.sidebar.info(
    "The trained diffusion model uses "
    "300 DDPM timesteps."
)

use_control = st.sidebar.checkbox(
    "Enable ControlNet-style conditioning",
    value=False
)


# ============================================================
# TEXT PROMPT
# ============================================================

prompt = st.text_input(
    "📝 Text Prompt",
    value="a handwritten digit seven"
)


# ============================================================
# CONTROLNET CONDITION
# ============================================================

control_image = None
control_label = None

if use_control:

    st.subheader(
        "🖼️ ControlNet Condition"
    )

    st.info(
        "ControlNet will automatically use an "
        "MNIST handwritten digit as the control image."
    )


    # --------------------------------------------------------
    # MNIST TRANSFORM
    # --------------------------------------------------------

    transform = transforms.Compose([

        transforms.Resize(
            (32, 32)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
    ])


    # --------------------------------------------------------
    # LOAD MNIST
    # --------------------------------------------------------

    mnist_dataset = datasets.MNIST(
        root=os.path.join(
            PROJECT_ROOT,
            "data"
        ),
        train=False,
        download=True,
        transform=transform
    )


    # --------------------------------------------------------
    # GET CONTROL IMAGE
    # --------------------------------------------------------

    control_image, control_label = (
        mnist_dataset[0]
    )


    # --------------------------------------------------------
    # DISPLAY CONTROL IMAGE
    # --------------------------------------------------------

    control_image_display = (

        control_image
        .squeeze(0)
        .cpu()
    )


    control_image_display = (

        control_image_display
        + 1.0
    ) / 2.0


    control_image_display = (

        control_image_display
        .clamp(
            0,
            1
        )
        .numpy()
    )


    st.image(
        control_image_display,
        caption=(
            "Automatic MNIST Control Digit: "
            f"{control_label}"
        ),
        width=200
    )


# ============================================================
# GENERATE BUTTON
# ============================================================

st.divider()

generate_button = st.button(
    "🚀 Generate Image",
    type="primary"
)


# ============================================================
# GENERATION
# ============================================================

if generate_button:

    if prompt.strip() == "":

        st.error(
            "Please enter a prompt."
        )

        st.stop()


    # ========================================================
    # CHECKPOINT VALIDATION
    # ========================================================

    required_files = [

        VAE_CHECKPOINT,

        CFG_CHECKPOINT,

        CONTROLNET_CHECKPOINT
    ]

    missing_files = []

    for file_path in required_files:

        if not os.path.exists(
            file_path
        ):

            missing_files.append(
                file_path
            )


    if missing_files:

        st.error(
            "The following checkpoint files are missing:"
        )

        for file_path in missing_files:

            st.code(
                file_path
            )

        st.stop()


    try:

        # ====================================================
        # LOAD MODELS
        # ====================================================

        with st.spinner(
            "Loading trained models..."
        ):

            (
                vae,
                model,
                text_encoder,
                controlnet
            ) = load_models()


        # ====================================================
        # CONTROLNET LATENT
        # ====================================================

        control_latent = None

        if use_control:

            with st.spinner(
                "Encoding MNIST control image..."
            ):

                control_latent = image_to_latent(
                    vae,
                    control_image
                )

            st.success(
                "ControlNet condition ready. "
                f"MNIST digit: {control_label}"
            )


        # ====================================================
        # START GENERATION
        # ====================================================

        st.info(
            "Running 300-step DDPM reverse diffusion..."
        )

        progress_bar = st.progress(
            0
        )


        result = generate_image(
            prompt=prompt,
            guidance_scale=guidance_scale,
            control_latent=control_latent,
            progress_bar=progress_bar
        )


        progress_bar.progress(
            100
        )


        # ====================================================
        # CONVERT OUTPUT TO IMAGE
        # ====================================================

        result_array = (

            result[0, 0]
            .detach()
            .cpu()
            .numpy()
        )


        result_array = (

            result_array
            * 255.0
        )


        result_array = (

            result_array
            .clip(
                0,
                255
            )
            .astype(
                "uint8"
            )
        )


        result_image = Image.fromarray(
            result_array
        )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.success(
            "🎉 Generation complete!"
        )

        st.image(
            result_image,
            caption="Generated Image",
            width=320
        )


        # ====================================================
        # SAVE OUTPUT
        # ====================================================

        output_path = os.path.join(
            PROJECT_ROOT,
            "outputs",
            "streamlit_generated.png"
        )


        result_image.save(
            output_path
        )


        # ====================================================
        # DOWNLOAD BUTTON
        # ====================================================

        with open(
            output_path,
            "rb"
        ) as file:

            st.download_button(
                label="⬇️ Save Generated Image",
                data=file,
                file_name="generated_image.png",
                mime="image/png"
            )


    except Exception as error:

        st.error(
            "❌ Generation failed."
        )

        st.exception(
            error
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "ℹ️ Model Information"
):

    st.write(
        "### Architecture"
    )

    st.write(
        "• Spatial VAE: 32×32 → 4×8×8 latent"
    )

    st.write(
        "• Transformer Text Encoder"
    )

    st.write(
        "• Text-conditioned latent U-Net"
    )

    st.write(
        "• Cross-Attention"
    )

    st.write(
        "• Classifier-Free Guidance"
    )

    st.write(
        "• ControlNet-style conditioning"
    )

    st.write(
        "• 300-step DDPM sampling"
    )

    st.write(
        "### Device"
    )

    st.code(
        DEVICE
    )

    if DEVICE == "cuda":

        st.write(
            torch.cuda.get_device_name(0)
        )

    st.warning(
        "This is an educational Stable-Diffusion-style "
        "MNIST system, not a general-purpose text-to-image model."
    )