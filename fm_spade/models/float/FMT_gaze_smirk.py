import os, math, torch
from weakref import ref
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint
from models import BaseModel

from timm.layers import use_fused_attn
from timm.models.vision_transformer import Mlp


def enc_dec_mask(T, S, frame_width = 1, expansion = 2):
	mask = torch.ones(T, S)
	for i in range(T):
		mask[i, max(0, (i - expansion) * frame_width):(i + expansion + 1) * frame_width] = 0
	return mask == 1


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
	"""
	Sinusoidal position encoding table.
	Args:
		n_position (int): the length of the input sequence
		d_hid (int): the dimension of the hidden state
	"""
	def cal_angle(position, hid_idx):
		return position / (10000 ** (2 * (hid_idx // 2) / d_hid))

	def get_posi_angle_vec(position):
		return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

	sinusoid_table = torch.Tensor([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
	sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
	sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1
	if padding_idx is not None: sinusoid_table[padding_idx] = 0.
	return sinusoid_table


class Attention(nn.Module):
	def __init__(
			self,
			dim: int,
			num_heads: int = 8,
			qkv_bias: bool = False,
			qk_norm: bool = False,
			attn_drop: float = 0.,
			proj_drop: float = 0.,
			norm_layer: nn.Module = nn.LayerNorm,
	) -> None:

		super().__init__()
		assert dim % num_heads == 0, 'dim should be divisible by num_heads'
		self.num_heads = num_heads
		self.head_dim = dim // num_heads
		self.scale = self.head_dim ** -0.5
		self.fused_attn = use_fused_attn()

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
		self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
		B, N, C = x.shape
		
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
		q, k, v = qkv.unbind(0)
		q, k = self.q_norm(q), self.k_norm(k)

		if self.fused_attn:
			x = F.scaled_dot_product_attention(
				q, k, v,
				attn_mask = ~mask,
				dropout_p=self.attn_drop.p if self.training else 0.,
			)
		else:
			q = q * self.scale
			attn = q @ k.transpose(-2, -1)
			attn = attn.softmax(dim=-1)
			attn = self.attn_drop(attn)
			x = attn @ v

		x = x.transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x

class TimestepEmbedder(nn.Module):
	"""
	Embeds scalar timesteps into vector representations.
	"""
	def __init__(self, hidden_size, frequency_embedding_size = 256):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(frequency_embedding_size, hidden_size, bias=True),
			nn.SiLU(),
			nn.Linear(hidden_size, hidden_size, bias=True),
		)
		self.frequency_embedding_size = frequency_embedding_size

	@staticmethod
	def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
		"""
		Create sinusoidal timestep embeddings.
		:param t: a 1-D Tensor of N indices, one per batch element.
						  These may be fractional.
		:param dim: the dimension of the output.
		:param max_period: controls the minimum frequency of the embeddings.
		:return: an (N, D) Tensor of positional embeddings.
		"""
		# https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
		half = dim // 2
		freqs = torch.exp(
			-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
		).to(device=t.device)
		args = t[:, None].float() * freqs[None]
		embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
		if dim % 2:
			embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
		return embedding

	def forward(self, t: torch.Tensor) -> torch.Tensor:
		t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
		t_emb = self.mlp(t_freq)
		return t_emb

class SequenceEmbed(nn.Module):
	def __init__(
			self,
			dim_w,
			dim_h,
			norm_layer=None,
			bias=True,
	):
		super().__init__()

		self.proj = nn.Linear(dim_w, dim_h, bias=bias)
		self.norm = norm_layer(dim_h) if norm_layer else nn.Identity()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.norm(self.proj(x))


class FMTBlock(nn.Module):
	"""
	A FMT block inspried by DiT Block
	"""
	def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs) -> None:
		super().__init__()
		self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
		self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
		self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
		mlp_hidden_dim = int(hidden_size * mlp_ratio)
		approx_gelu = lambda: nn.GELU(approximate="tanh")
		self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
		self.adaLN_modulation = nn.Sequential(
			nn.SiLU(),
			nn.Linear(hidden_size, 6 * hidden_size, bias=True)
		)

	def framewise_modulate(self, x, shift, scale) -> torch.Tensor:
		return x * (1 + scale) + shift

	def forward(self, x, c, mask=None) -> torch.Tensor:
		assert mask is not None
		shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)        
		x = x + gate_msa * self.attn(self.framewise_modulate(self.norm1(x), shift_msa, scale_msa), mask = mask)
		x = x + gate_mlp * self.mlp(self.framewise_modulate(self.norm2(x), shift_mlp, scale_mlp))
		return x

class Decoder(nn.Module):
	"""
	The final decoder of FlowMatchingTransformer.
	"""
	def __init__(self, hidden_size, dim_w):
		super().__init__()
		self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
		self.adaLN_modulation = nn.Sequential(
			nn.SiLU(),
			nn.Linear(hidden_size, 2 * hidden_size, bias=True)
		)
		self.linear = nn.Linear(hidden_size, dim_w, bias=True)

	def framewise_modulate(self, x, shift, scale) -> torch.Tensor:
		return x * (1 + scale) + shift

	def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
		shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
		x = self.framewise_modulate(self.norm_final(x), shift, scale)
		return self.linear(x)


class FlowMatchingTransformer(BaseModel):
    """
    Flow Matching Transformer (FMT)
    """
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

        self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(opt.num_prev_frames)
        self.num_total_frames = self.num_prev_frames + self.num_frames_for_clip

        self.hidden_size = opt.dim_h
        self.mlp_ratio = opt.mlp_ratio
        self.fmt_depth = opt.fmt_depth
        self.num_heads = opt.num_heads

        # 输入序列嵌入
        self.x_embedder = SequenceEmbed(2 * opt.dim_motion, self.hidden_size)

        # video time position encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_total_frames, self.hidden_size),
            requires_grad=False
        )

        # flow trajectory time encoding
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.c_embedder = nn.Linear(opt.dim_c, self.hidden_size)

        # define FMT blocks
        self.blocks = nn.ModuleList([
            FMTBlock(self.hidden_size, self.num_heads, mlp_ratio=self.mlp_ratio)
            for _ in range(self.fmt_depth)
        ])
        self.decoder = Decoder(self.hidden_size, self.opt.dim_motion)
        self.initialize_weights()

        # define alignment mask
        alignment_mask = enc_dec_mask(
            self.num_total_frames, self.num_total_frames, 1,
            expansion=opt.attention_window
        )
        self.register_buffer("alignment_mask", alignment_mask)

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_sinusoid_encoding_table(
            self.num_total_frames, self.hidden_size
        )
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in FMT blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.decoder.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.linear.weight, 0)
        nn.init.constant_(self.decoder.linear.bias, 0)

    def sequence_embedder(
        self, sequence: torch.Tensor,
        dropout_prob: float,
        train: bool = False
    ) -> torch.Tensor:
        if train:
            batch_id_for_drop = torch.where(
                torch.rand(sequence.shape[0], device=sequence.device) < dropout_prob
            )
            sequence[batch_id_for_drop] = 0
        return sequence

    def forward(
        self,
        t,
        x,
        a,
        prev_x,
        prev_a,
        ref_x,
        gaze,
        prev_gaze,
        pose,
        prev_pose,
        cam,
        prev_cam,
        train: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of Conditional Flow Matching Transformer.

        Args:
            t: (B,) tensor of diffusion timesteps [0, 1]
            x: (B, L, 512) motion latent
            a: (B, L, 512) audio latent
            pose: (B, L, 6) pose latent
            ref_x: (B, 512) reference motion latent
            prev_x, prev_a, prev_pose, ... : auto-regressive context

        Returns:
            (B, L, dim_motion) motion prediction
        """
        # time encoding
        t = self.t_embedder(t).unsqueeze(1)  # (N, D)
        # condition encoding
        a    = self.sequence_embedder(a,    dropout_prob=self.opt.audio_dropout_prob, train=train)
        pose = self.sequence_embedder(pose, dropout_prob=self.opt.audio_dropout_prob, train=train)
        cam  = self.sequence_embedder(cam,  dropout_prob=self.opt.audio_dropout_prob, train=train)
        gaze = self.sequence_embedder(gaze, dropout_prob=self.opt.audio_dropout_prob, train=train)

        if prev_x is not None:
            prev_x    = self.sequence_embedder(prev_x,    dropout_prob=0.5, train=train)
            prev_a    = self.sequence_embedder(prev_a,    dropout_prob=0.5, train=train)
            prev_pose = self.sequence_embedder(prev_pose, dropout_prob=0.5, train=train)
            prev_cam  = self.sequence_embedder(prev_cam,  dropout_prob=0.5, train=train)
            prev_gaze = self.sequence_embedder(prev_gaze, dropout_prob=0.5, train=train)

            x    = torch.cat([prev_x, x], dim=1)
            a    = torch.cat([prev_a, a], dim=1)
            pose = torch.cat([prev_pose, pose], dim=1)
            cam  = torch.cat([prev_cam, cam], dim=1)
            gaze = torch.cat([prev_gaze, gaze], dim=1)

        ref_x = ref_x[:, None, ...].repeat(1, x.shape[1], 1)
        x     = torch.cat([ref_x, x], dim=-1)
        x     = self.x_embedder(x)
        x     = x + self.pos_embed  # (N, L + L', D)

        c = self.c_embedder(a + pose + cam + gaze)
        c = t + c

        # forwarding FMT Blocks
        for block in self.blocks:
            x = block(x, c, self.alignment_mask)  # (N, T, D)

        return self.decoder(x, c)

    @torch.no_grad()
    def forward_with_cfv(
        self,
        t,
        x,
        a,
        prev_x,
        prev_a,
        ref_x,
        gaze,
        prev_gaze,
        pose,
        prev_pose,
        cam,
        prev_cam,
        a_cfg_scale: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with Classifier-Free Guidance (CFG).
        """
        if a_cfg_scale != 1.0:
            # null 条件，用于做 classifier-free guidance
            null_a    = torch.zeros_like(a)

            # 拼接 unconditional / conditional
            audio_cat     = torch.cat([null_a,    a],    dim=0)
            gaze_cat      = torch.cat([gaze, gaze], dim=0)
            pose_cat      = torch.cat([pose, pose], dim=0)
            cam_cat       = torch.cat([cam,  cam],  dim=0)

            x_cat         = torch.cat([x, x], dim=0)
            prev_x_cat    = torch.cat([prev_x, prev_x], dim=0)
            prev_a_cat    = torch.cat([prev_a, prev_a], dim=0)
            prev_gaze_cat = torch.cat([prev_gaze, prev_gaze], dim=0)
            prev_pose_cat = torch.cat([prev_pose, prev_pose], dim=0)
            prev_cam_cat  = torch.cat([prev_cam, prev_cam], dim=0)
            ref_x_cat     = torch.cat([ref_x, ref_x], dim=0)

            # forward with cond + uncond
            model_output = self.forward(
                t=t,
                x=x_cat,
                a=audio_cat,
                prev_x=prev_x_cat,
                prev_a=prev_a_cat,
                ref_x=ref_x_cat,
                gaze=gaze_cat,
                prev_gaze=prev_gaze_cat,
                pose=pose_cat,
                prev_pose=prev_pose_cat,
                cam=cam_cat,
                prev_cam=prev_cam_cat,
				train=False
            )

            # 拆分 uncond / cond
            uncond, all_cond = torch.chunk(model_output, chunks=2, dim=0)

            # classifier-free vector field
            return uncond + a_cfg_scale * (all_cond - uncond)

        else:
            return self.forward(
                t=t,
                x=x,
                a=a,
                prev_x=prev_x,
                prev_a=prev_a,
                ref_x=ref_x,
                gaze=gaze,
                prev_gaze=prev_gaze,
                pose=pose,
                prev_pose=prev_pose,
                cam=cam,
                prev_cam=prev_cam,
                train=False
            )
