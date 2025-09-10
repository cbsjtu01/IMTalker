from turtle import pos
import torch, math
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint
from transformers import Wav2Vec2Config
from transformers.modeling_outputs import BaseModelOutput

from models.wav2vec2 import Wav2VecModel
from models.wav2vec2_ser import Wav2Vec2ForSpeechClassification

from models import BaseModel
from models.float.FMT_gaze_smirk import FlowMatchingTransformer

######## Main Phase 2 model ########
class ConditionFMT(BaseModel):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        # 基本参数
        self.num_frames_for_clip = int(opt.wav2vec_sec * opt.fps)
        self.num_prev_frames = int(opt.num_prev_frames)
        self.num_total_frames = self.num_frames_for_clip + self.num_prev_frames
        self.audio_input_dim = 768 if opt.only_last_features else 12 * 768

        # 模块定义
        self.audio_encoder = AudioEncoder(opt)
        self.audio_projection = self._make_projection(self.audio_input_dim, opt.dim_c)
        self.gaze_projection  = self._make_projection(2, opt.dim_c)
        self.pose_projection  = self._make_projection(3, opt.dim_c)
        self.cam_projection   = self._make_projection(3, opt.dim_c)

        # Flow Matching Transformer
        self.fmt = FlowMatchingTransformer(opt)
        self.odeint_kwargs = {
            'atol': opt.ode_atol,
            'rtol': opt.ode_rtol,
            'method': opt.torchdiffeq_ode_method
        }

    def _make_projection(self, in_dim, out_dim):
        """创建通用投影模块"""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU()
        )

    def forward(self, batch, t):
        # 解包 batch
        x, prev_x, a, prev_a, m_ref = batch["m_now"], batch["m_prev"], batch["a_now"], batch["a_prev"], batch["m_ref"]
        gaze, prev_gaze = batch["gaze"], batch["gaze_prev"]
        pose, prev_pose = batch["pose"], batch["pose_prev"]
        cam, prev_cam   = batch["cam"], batch["cam_prev"]

        bs = x.size(0)

        # reshape 多层音频或条件特征
        if not self.opt.only_last_features:
            a         = a.view(bs, self.num_frames_for_clip, -1)
            prev_a    = prev_a.view(bs, self.num_prev_frames, -1)

        # 投影
        a         = self.audio_projection(a)
        prev_a    = self.audio_projection(prev_a)
        gaze      = self.gaze_projection(gaze)
        prev_gaze = self.gaze_projection(prev_gaze)
        pose      = self.pose_projection(pose)
        prev_pose = self.pose_projection(prev_pose)
        cam       = self.cam_projection(cam)
        prev_cam  = self.cam_projection(prev_cam)

        # 将所有条件融合到 FMT
        # 假设 FMT 可以接受额外条件参数（你需要在 FMT 里实现）
        pred = self.fmt(
            t,
            x.squeeze(),
            a,
            prev_x,
            prev_a,
            m_ref,
            gaze=gaze,
            prev_gaze=prev_gaze,
            pose=pose,
            prev_pose=prev_pose,
            cam=cam,
            prev_cam=prev_cam
        )

        # 返回去掉 prev 部分的预测
        return pred[:, self.num_prev_frames:, ...]

    @torch.no_grad()
    def sample(
        self,
        data: dict,
        a_cfg_scale: float = 1.0,
        nfe: int = 10,
        seed: int = None
    ) -> torch.Tensor:
        
        a, ref_x = data['a'], data['ref_x']
        gaze, pose, cam = data.get('gaze', None), data.get('pose', None), data.get('cam', None)
        B = a.shape[0]
    
        # make time
        time = torch.linspace(0, 1, nfe, device=self.opt.rank)
        
        # audio encode
        a = a.to(self.opt.rank)
        T = math.ceil(a.shape[-1] * self.opt.fps / self.opt.sampling_rate)
        a = self.audio_encoder.inference(a, seq_len=T)
        a = self.audio_projection(a)
    
        # ============ 条件投影 ============ #
        if gaze is not None:
            gaze = self.gaze_projection(gaze.to(self.opt.rank)).unsqueeze(0)
        else:
            gaze = torch.zeros(B, T, self.opt.dim_c, device=self.opt.rank)
    
        if pose is not None:
            pose = self.pose_projection(pose.to(self.opt.rank)).unsqueeze(0)
        else:
            pose = torch.zeros(B, T, self.opt.dim_c, device=self.opt.rank)
    
        if cam is not None:
            cam = self.cam_projection(cam.to(self.opt.rank)).unsqueeze(0)
        else:
            cam = torch.zeros(B, T, self.opt.dim_c, device=self.opt.rank)
        # ================================= #
    
        sample = []
        for t in range(0, int(math.ceil(T / self.num_frames_for_clip))):
            if self.opt.fix_noise_seed:
                seed = self.opt.seed if seed is None else seed
                g = torch.Generator(self.opt.rank)
                g.manual_seed(seed)
                x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device=self.opt.rank, generator=g)
            else:
                x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device=self.opt.rank)
    
            if t == 0:  # 初始化 prev
                prev_x_t    = torch.zeros(B, self.num_prev_frames, self.opt.dim_c).to(self.opt.rank)
                prev_a_t    = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
                prev_gaze_t = torch.zeros(B, self.num_prev_frames, gaze.shape[-1]).to(self.opt.rank)
                prev_pose_t = torch.zeros(B, self.num_prev_frames, pose.shape[-1]).to(self.opt.rank)
                prev_cam_t  = torch.zeros(B, self.num_prev_frames, cam.shape[-1]).to(self.opt.rank)
            else:
                prev_x_t    = sample_t[:, -self.num_prev_frames:]
                prev_a_t    = a_t[:, -self.num_prev_frames:]
                prev_gaze_t = gaze_t[:, -self.num_prev_frames:]
                prev_pose_t = pose_t[:, -self.num_prev_frames:]
                prev_cam_t  = cam_t[:, -self.num_prev_frames:]
    
            # 当前 clip 的切片
            a_t    = a[:,    t*self.num_frames_for_clip:(t+1)*self.num_frames_for_clip]
            gaze_t = gaze[:, t*self.num_frames_for_clip:(t+1)*self.num_frames_for_clip]
            pose_t = pose[:, t*self.num_frames_for_clip:(t+1)*self.num_frames_for_clip]
            cam_t  = cam[:,  t*self.num_frames_for_clip:(t+1)*self.num_frames_for_clip]
    
            # padding 处理
            for cond_name, cond_tensor in [
                ("a_t", a_t), ("gaze_t", gaze_t), ("pose_t", pose_t), ("cam_t", cam_t)
            ]:
                if cond_tensor.dim() == 3:  # (B, T, C)
                    if cond_tensor.shape[1] < self.num_frames_for_clip:
                        pad_len = self.num_frames_for_clip - cond_tensor.shape[1]
                        last_frame = cond_tensor[:, -1:, :].expand(-1, pad_len, -1)
                        cond_tensor = torch.cat([cond_tensor, last_frame], dim=1)
    
                if cond_name == "a_t": a_t = cond_tensor
                elif cond_name == "gaze_t": gaze_t = cond_tensor
                elif cond_name == "pose_t": pose_t = cond_tensor
                elif cond_name == "cam_t": cam_t = cond_tensor
    
            # 定义 ODE 系统
            def sample_chunk(tt, zt):
                out = self.fmt.forward_with_cfv(
                    t           = tt.unsqueeze(0),
                    x           = zt,
                    a           = a_t,
                    prev_x      = prev_x_t,
                    prev_a      = prev_a_t,
                    ref_x       = ref_x,
                    gaze        = gaze_t,
                    prev_gaze   = prev_gaze_t,
                    pose        = pose_t,
                    prev_pose   = prev_pose_t,
                    cam         = cam_t,
                    prev_cam    = prev_cam_t,
                    a_cfg_scale = a_cfg_scale,
                )
                return out[:, self.num_prev_frames:]
    
            # solve ODE
            trajectory_t = odeint(sample_chunk, x0, time, **self.odeint_kwargs)
            sample_t = trajectory_t[-1]
            sample.append(sample_t)
    
        sample = torch.cat(sample, dim=1)[:, :T]
        return sample



################# Condition Encoders ################
class AudioEncoder(BaseModel):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.only_last_features = opt.only_last_features
        
        self.num_frames_for_clip = int(opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(opt.num_prev_frames)

        self.wav2vec2 = Wav2VecModel.from_pretrained(opt.wav2vec_model_path, local_files_only = True)
        self.wav2vec2.feature_extractor._freeze_parameters()

        for name, param in self.wav2vec2.named_parameters():
            param.requires_grad = False

    def get_wav2vec2_feature(self, a: torch.Tensor, seq_len:int) -> torch.Tensor:
        a = self.wav2vec2(a, seq_len=seq_len, output_hidden_states = not self.only_last_features)
        if self.only_last_features:
            a = a.last_hidden_state
        else:
            a = torch.stack(a.hidden_states[1:], dim=1).permute(0, 2, 1, 3)
            a = a.reshape(a.shape[0], a.shape[1], -1)
        return a

    def forward(self, a:torch.Tensor, prev_a:torch.Tensor = None) -> torch.Tensor:
        if prev_a is not None:
            a = torch.cat([prev_a, a], dim = 1)
            if a.shape[1] % int( (self.num_frames_for_clip + self.num_prev_frames) * self.opt.sampling_rate / self.opt.fps) != 0:
                a = F.pad(a, (0, int((self.num_frames_for_clip + self.num_prev_frames) * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode='replicate')
            a = self.get_wav2vec2_feature(a, seq_len = self.num_frames_for_clip + self.num_prev_frames)
        else:
            if a.shape[1] % int( self.num_frames_for_clip * self.opt.sampling_rate / self.opt.fps) != 0:
                a = F.pad(a, (0, int(self.num_frames_for_clip * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode = 'replicate')
            a = self.get_wav2vec2_feature(a, seq_len = self.num_frames_for_clip)
    
        return a

    @torch.no_grad()
    def inference(self, a: torch.Tensor, seq_len:int) -> torch.Tensor:
        if a.shape[1] % int(seq_len * self.opt.sampling_rate / self.opt.fps) != 0:
            a = F.pad(a, (0, int(seq_len * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode = 'replicate')
        a = self.get_wav2vec2_feature(a, seq_len=seq_len)
        return a