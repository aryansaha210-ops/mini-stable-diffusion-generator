import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TIMESTEPS = 300

GUIDANCE_SCALE = 5.0

CFG_CHECKPOINT = (
    "checkpoints/cfg_text_real_latent_epoch_5.pth"
)

CONTROLNET_CHECKPOINT = (
    "checkpoints/controlnet_epoch_5.pth"
)

VAE_CHECKPOINT = (
    "checkpoints/spatial_vae_epoch_10.pth"
)


# ============================================================
# SPATIAL VAE
# ============================================================

class SpatialVAE(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1, 32, 4, 2, 1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32, 64, 4, 2, 1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64, 128, 3, 1, 1
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
# TOKENIZER
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


def tokenize(text):

    words = text.lower().strip().split()

    tokens = [START_TOKEN]

    for word in words:

        tokens.append(
            VOCAB.get(
                word,
                UNK_TOKEN
            )
        )

    tokens.append(END_TOKEN)

    tokens = tokens[:MAX_TOKENS]

    while len(tokens) < MAX_TOKENS:

        tokens.append(PAD_TOKEN)

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
            10000,
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
            x.flatten(2)
            .transpose(1, 2)
        )

        Q = self.query(x_flat)

        K = self.key(text)

        V = self.value(text)

        Q = Q.view(
            B,
            -1,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            B,
            -1,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            B,
            -1,
            self.heads,
            self.head_dim
        ).transpose(1, 2)

        attention = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        attention = (
            attention /
            (self.head_dim ** 0.5)
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
            result.transpose(1, 2)
            .contiguous()
        )

        result = result.view(
            B,
            -1,
            128
        )

        result = self.output(result)

        result = (
            result.transpose(1, 2)
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

        return self.network(t)


# ============================================================
# RESBLOCK
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

        h = self.conv1(x)

        h = (
            h
            + self.time_projection(
                time_embedding
            ).unsqueeze(-1).unsqueeze(-1)
        )

        h = (
            h
            + self.text_projection(
                text_embedding
            ).unsqueeze(-1).unsqueeze(-1)
        )

        h = F.silu(h)

        h = self.conv2(h)

        h = F.silu(h)

        return h + self.skip(x)


# ============================================================
# CFG U-NET
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

        x0 = self.input(x)

        x1 = self.res1(
            x0,
            time_embedding,
            text_mean
        )

        x2 = self.down(x1)

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

        x4 = self.up(x3)

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

        return self.output(x4)


# ============================================================
# CONTROLNET
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

            # 8x8 -> 4x4
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

    def forward(self, condition):

        x = self.condition_encoder(
            condition
        )

        x = self.zero_conv(x)

        return x


# ============================================================
# LOAD VAE
# ============================================================

print()
print("Loading SpatialVAE...")

vae = SpatialVAE().to(
    DEVICE
)

vae_checkpoint = torch.load(
    VAE_CHECKPOINT,
    map_location=DEVICE
)

if "model" in vae_checkpoint:

    vae.load_state_dict(
        vae_checkpoint["model"]
    )

else:

    vae.load_state_dict(
        vae_checkpoint
    )

vae.eval()

print("SpatialVAE loaded.")


# ============================================================
# LOAD CFG MODEL
# ============================================================

print()
print("Loading CFG diffusion model...")

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

print("CFG model loaded.")


# ============================================================
# LOAD CONTROLNET
# ============================================================

print()
print("Loading ControlNet...")

controlnet = ControlNetCondition().to(
    DEVICE
)

control_checkpoint = torch.load(
    CONTROLNET_CHECKPOINT,
    map_location=DEVICE
)

controlnet.load_state_dict(
    control_checkpoint["model"]
)

controlnet.eval()

print("ControlNet loaded.")


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
# GENERATION FUNCTION
# ============================================================

@torch.no_grad()
def generate(
    prompt,
    condition_latent
):

    # --------------------------------------------------------
    # Tokenize prompt
    # --------------------------------------------------------

    tokens = tokenize(
        prompt
    )

    # --------------------------------------------------------
    # Encode text
    # --------------------------------------------------------

    text_features = (
        text_encoder(tokens)
    )

    # --------------------------------------------------------
    # Unconditional text
    # --------------------------------------------------------

    unconditional_tokens = tokenize(
        ""
    )

    unconditional_features = (
        text_encoder(
            unconditional_tokens
        )
    )

    # --------------------------------------------------------
    # ControlNet condition
    # --------------------------------------------------------

    control_features = controlnet(
        condition_latent
    )

    # --------------------------------------------------------
    # Start from random noise
    # --------------------------------------------------------

    latent = torch.randn(
        1,
        4,
        8,
        8,
        device=DEVICE
    )

    print()
    print("Starting reverse diffusion...")

    # --------------------------------------------------------
    # Reverse diffusion
    # --------------------------------------------------------

    for timestep in reversed(
        range(TIMESTEPS)
    ):

        t = torch.tensor(
            [timestep],
            device=DEVICE
        )

        # ----------------------------------------------------
        # Conditional prediction
        # ----------------------------------------------------

        conditional_noise = model(
            latent,
            t,
            text_features,
            control=control_features
        )

        # ----------------------------------------------------
        # Unconditional prediction
        # ----------------------------------------------------

        unconditional_noise = model(
            latent,
            t,
            unconditional_features,
            control=control_features
        )

        # ----------------------------------------------------
        # Classifier-Free Guidance
        # ----------------------------------------------------

        guided_noise = (
            unconditional_noise
            +
            GUIDANCE_SCALE
            * (
                conditional_noise
                -
                unconditional_noise
            )
        )

        # ----------------------------------------------------
        # DDPM reverse step
        # ----------------------------------------------------

        beta_t = betas[timestep]

        alpha_t = alphas[timestep]

        alpha_bar_t = alpha_bars[timestep]

        latent = (
            1.0
            / torch.sqrt(alpha_t)
        ) * (
            latent
            -
            (
                beta_t
                /
                torch.sqrt(
                    1.0 - alpha_bar_t
                )
            )
            * guided_noise
        )

        # ----------------------------------------------------
        # Add noise except at final step
        # ----------------------------------------------------

        if timestep > 0:

            noise = torch.randn_like(
                latent
            )

            latent = (
                latent
                +
                torch.sqrt(beta_t)
                * noise
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            timestep % 50 == 0
            or timestep == TIMESTEPS - 1
        ):

            print(
                "Reverse step:",
                timestep
            )

    # --------------------------------------------------------
    # Decode latent
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

    return image


# ============================================================
# MAIN
# ============================================================

print()
print("Device:", DEVICE)
print()

print(
    "ControlNet generation is ready."
)

print()
print(
    "For this educational ControlNet setup,"
)
print(
    "we use an MNIST image as the control condition."
)
print()

# ------------------------------------------------------------
# Load one MNIST image
# ------------------------------------------------------------

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

dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

condition_image, label = dataset[0]

condition_image = (
    condition_image
    .unsqueeze(0)
    .to(DEVICE)
)

# ------------------------------------------------------------
# Convert condition image to latent
# ------------------------------------------------------------

with torch.no_grad():

    condition_mu, condition_logvar = (
        vae.encode(
            condition_image
        )
    )

    condition_latent = condition_mu

# ------------------------------------------------------------
# Ask for prompt
# ------------------------------------------------------------

prompt = input(
    "Enter prompt "
    "(example: a handwritten digit seven): "
)

if prompt.strip() == "":

    prompt = (
        "a handwritten digit "
        + str(label)
    )

print()
print("Prompt:", prompt)

print(
    "Control digit:",
    label
)

# ------------------------------------------------------------
# Generate
# ------------------------------------------------------------

generated_image = generate(
    prompt,
    condition_latent
)

# ------------------------------------------------------------
# Save image
# ------------------------------------------------------------

output_path = (
    "outputs/controlnet_generated.png"
)

generated_image = (
    generated_image
    .squeeze(0)
    .squeeze(0)
    .cpu()
)

plt.figure(
    figsize=(4, 4)
)

plt.imshow(
    generated_image,
    cmap="gray"
)

plt.axis("off")

plt.title(
    "ControlNet Generation"
)

plt.tight_layout()

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print()
print(
    "Generated image saved:"
)

print(
    output_path
)

print()
print("CONTROLNET GENERATION DONE")