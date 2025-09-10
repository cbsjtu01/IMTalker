"""
	Inference Stage 2
"""

import os, torch, random, cv2, torchvision, subprocess, librosa, datetime, tempfile, face_alignment
import numpy as np
import albumentations as A
import albumentations.pytorch.transforms as A_pytorch

from tqdm import tqdm
from pathlib import Path
from traitlets import default
from transformers import Wav2Vec2FeatureExtractor

from models.float.CFMT_gaze_smirk import ConditionFMT
from models.networks.model_bestdecoder import IMFModel
from options.base_options import BaseOptions
from PIL import Image
import torchvision.transforms as transforms

def load_smirk(smirk):
	pose = smirk["pose_params"].cuda()  # (1, 3)
	cam = smirk["cam"].cuda()                  # (1, 3)
	return pose, cam


class DataProcessor:
	def __init__(self, opt):
		self.opt = opt
		self.fps = opt.fps
		self.sampling_rate = opt.sampling_rate
		self.input_size = opt.input_size

		self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

		# wav2vec2 audio preprocessor
		self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(opt.wav2vec_model_path, local_files_only=True)

		# image transform 
		self.transform = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
        ])

	@torch.no_grad()
	def process_img(self, img:np.ndarray) -> np.ndarray:
		mult = 360. / img.shape[0]

		resized_img = cv2.resize(img, dsize=(0, 0), fx = mult, fy = mult, interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)        
		bboxes = self.fa.face_detector.detect_from_image(resized_img)
		bboxes = [(int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score) for (x1, y1, x2, y2, score) in bboxes if score > 0.95]
		bboxes = bboxes[0] # Just use first bbox

		bsy = int((bboxes[3] - bboxes[1]) / 2)
		bsx = int((bboxes[2] - bboxes[0]) / 2)
		my  = int((bboxes[1] + bboxes[3]) / 2)
		mx  = int((bboxes[0] + bboxes[2]) / 2)
		
		bs = int(max(bsy, bsx) * 1.6)
		img = cv2.copyMakeBorder(img, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
		my, mx  = my + bs, mx + bs  	# BBox center y, bbox center x
		
		crop_img = img[my - bs:my + bs,mx - bs:mx + bs]
		crop_img = cv2.resize(crop_img, dsize = (self.input_size, self.input_size), interpolation = cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
		return crop_img

	def default_img_loader(self, path) -> np.ndarray:
		img = cv2.imread(path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		return Image.fromarray(img)

	def default_aud_loader(self, path: str) -> torch.Tensor:
		speech_array, sampling_rate = librosa.load(path, sr = self.sampling_rate)
		return self.wav2vec_preprocessor(speech_array, sampling_rate = sampling_rate, return_tensors = 'pt').input_values[0]


	def preprocess(self, ref_path:str, audio_path:str, no_crop:bool) -> dict:
		s = self.default_img_loader(ref_path)
		if not no_crop:
			s = self.process_img(s)
		s = self.transform(s).unsqueeze(0)
		#import pdb; pdb.set_trace()
		a = self.default_aud_loader(audio_path).unsqueeze(0)
		return {'s': s, 'a': a, 'p': None, 'e': None}


class InferenceAgent:
	def __init__(self, opt):
		torch.cuda.empty_cache()
		self.opt = opt
		self.rank = opt.rank
		
		# Load Model
		self.load_model()
		self.load_weight(opt.fm_path, self.rank)
		self.ae.to(self.rank)
		self.fm.to(self.rank)
		self.ae.eval()
		self.fm.eval()

		# Load Data Processor
		self.data_processor = DataProcessor(opt)

	def load_model(self) -> None:
		self.ae = IMFModel()
		self.fm = ConditionFMT(self.opt)
		#import pdb;pdb.set_trace()
		motion_audioencoder_state_dict = torch.load(opt.imf_path, map_location="cpu")["state_dict"]
		ae_state_dict = {k.replace("gen.", ""): v for k, v in motion_audioencoder_state_dict.items() if k.startswith("gen.")}
		missing_gen, unexpected_gen = self.ae.load_state_dict(ae_state_dict, strict=False)
		#flow_matching_state_dict = torch.load(opt.fm_path, map_location="cpu")["state_dict"]
		#fm_state_dict = {k.replace("model.", ""): v for k, v in flow_matching_state_dict.items() if k.startswith("model.")}
		#missing_gen, unexpected_gen = self.fm.load_state_dict(fm_state_dict, strict=False)

	def load_weight(self, checkpoint_path: str, rank: int, prefix: str = 'model.') -> None:
		# 加载权重
		state_dict = torch.load(checkpoint_path, map_location='cpu')["state_dict"]
		if 'model' in state_dict:  # 如果有包装
			state_dict = state_dict['model']

		# 去掉 prefix 前缀
		stripped_state_dict = {
			k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
		}
		#import pdb;pdb.set_trace()
		with torch.no_grad():
			for model_name, param in self.fm.named_parameters():
				if model_name in stripped_state_dict:
					param.copy_(stripped_state_dict[model_name].to(rank))
				elif "wav2vec2" in model_name:
					pass  # 可选地跳过 wav2vec2 相关参数
				else:
					print(f"! Warning: {model_name} not found in state_dict.")

		del state_dict

	def save_video(self, vid_target_recon: torch.Tensor, video_path: str, audio_path: str) -> str:
		with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
			temp_filename = temp_video.name
			vid = vid_target_recon.permute(0, 2, 3, 1)
			vid = vid.detach().clamp(-1, 1).cpu()
			vid = (vid * 255).type('torch.ByteTensor')
			torchvision.io.write_video(temp_filename, vid, fps=self.opt.fps)			
			if audio_path is not None:
				with open(os.devnull, 'wb') as f:
					command =  "ffmpeg -i {} -i {} -c:v copy -c:a aac {} -y".format(temp_filename, audio_path, video_path)
					subprocess.call(command, shell=True, stdout=f, stderr=f)
			else:
				os.rename(temp_filename, video_path)
		if os.path.exists(video_path):
			os.remove(temp_filename)
		return video_path

	@torch.no_grad()
	def run_inference(
		self,
		res_video_path: str,
		ref_path: str,
		audio_path: str,
		pose_path: str = None,
		gaze_path: str = None,
		a_cfg_scale: float	= 1.0,
		nfe: int			= 10,
		no_crop: bool 		= False,
		seed: int			= 25,
		verbose: bool 		= False
	) -> str:

		data = self.data_processor.preprocess(ref_path, audio_path, no_crop = no_crop)
		# 如果有 pose 文件
		if pose_path is not None and os.path.exists(pose_path):
			data["pose"], data["cam"] = load_smirk(torch.load(pose_path))
		else:
			data["pose"], data["cam"] = None, None
		if verbose:
			print(f"> [Info] No pose provided for {ref_path}")
		# 如果有 gaze 文件
		if gaze_path is not None and os.path.exists(gaze_path):
			data["gaze"] = torch.tensor(np.load(gaze_path), dtype=torch.float32).cuda()
		else:
			data["gaze"] = None
			if verbose:
				print(f"> [Info] No gaze provided for {ref_path}")

		if verbose: print(f"> [Done] Preprocess.")

		f_r, t_r = self.encode_image_into_latent(data['s'].to(self.opt.rank))
		data["ref_x"] = t_r

		sample = self.fm.sample(data, a_cfg_scale = a_cfg_scale,  nfe = nfe, seed = seed)
		
		data_out = self.decode_latent_into_image(f_r = f_r, t_r = t_r, t_c = sample)
		
		res_video_path = self.save_video(data_out["d_hat"], res_video_path, audio_path)
		if verbose: print(f"> [Done] result saved at {res_video_path}")
		return res_video_path
	
	######## Motion Encoder - Decoder ########
	@torch.no_grad()
	def encode_image_into_latent(self, x: torch.Tensor) -> list:
		f = self.ae.dense_feature_encoder(x)
		t = self.ae.latent_token_encoder(x)
		return f, t
	
	def decode_latent_into_motion(self, t: torch.Tensor):
		return self.ae.latent_token_decoder(t)
	
	@torch.no_grad()
	def decode_latent_into_image(self, f_r: torch.Tensor , t_r: torch.Tensor, t_c: torch.Tensor) -> dict:
		B, T = t_c.shape[0], t_c.shape[1]
		m_r = self.decode_latent_into_motion(t_r)
		d_hat = []
		for t in range(T):
			m_c = self.decode_latent_into_motion(t_c[:,t,...])
			img = self.ae.ima(m_c, m_r, f_r)
			d_hat.append(img)
		d_hat = torch.stack(d_hat, dim=1).squeeze()
		return {'d_hat': d_hat}


class InferenceOptions(BaseOptions):
	def __init__(self):
		super().__init__()

	def initialize(self, parser):
		super().initialize(parser)
		parser.add_argument("--ref_path",
				default=None, type=str,help='ref')
		parser.add_argument("--pose_path",
				default=None, type=str,help='ref')
		parser.add_argument("--gaze_path",
				default=None, type=str,help='ref')
		parser.add_argument('--aud_path',
				default=None, type=str, help='audio')
		parser.add_argument('--no_crop',
				action = 'store_true', help = 'not using crop')
		parser.add_argument('--res_video_path',
				default=None, type=str, help='res video path')
		parser.add_argument('--fm_path',
				default="/home/nvadmin/workspace/taek/float-pytorch/checkpoints/float.pth", type=str, help='checkpoint path')
		parser.add_argument('--res_dir', default="./results", type=str, help='result dir')
		parser.add_argument('--input_root', default=None, type=str, help='input dir containing subdirs of ref + audio')
		parser.add_argument('--imf_path', default="E:\codes\codes\IMF_last\exps\\0.2_vgg\checkpoints\last.ckpt", type=str)
		parser.add_argument("--dim_motion", type=int, default=32)
		parser.add_argument("--dim_c", type=int, default=32)
		parser.add_argument('--dim_w', type=int, default=32, help='face dimension')
		return parser


if __name__ == '__main__':
    import os
    import datetime

    opt = InferenceOptions().parse()
    opt.rank, opt.ngpus = 0, 1
    agent = InferenceAgent(opt)
    os.makedirs(opt.res_dir, exist_ok=True)

    # 指定输入的根目录（包含多个子目录，每个子目录有图像和音频）
    input_root = opt.input_root  # 新增参数，比如：--input_root data/test_videos
    pose_root = getattr(opt, "pose_path", None)   # 可能不存在
    gaze_root = getattr(opt, "gaze_path", None)   # 可能不存在

    # 遍历所有子目录
    for subdir in sorted(os.listdir(input_root)):
        sub_path = os.path.join(input_root, subdir)
        if not os.path.isdir(sub_path):
            continue

        # 寻找图像和音频文件
        ref_path, aud_path = None, None
        for file in os.listdir(sub_path):
            if file.lower().endswith(('.jpg', '.png')):
                ref_path = os.path.join(sub_path, file)
            elif file.lower().endswith(('.wav', '.m4a', '.mp3')):
                aud_path = os.path.join(sub_path, file)

        filename_wo_ext = os.path.splitext(subdir)[0]  # 得到 "video"

        # pose/gaze 可选
        pose_path = None
        gaze_path = None
        if pose_root is not None:
            cand_pose = os.path.join(pose_root, filename_wo_ext + ".pt")
            if os.path.exists(cand_pose):
                pose_path = cand_pose
        if gaze_root is not None:
            cand_gaze = os.path.join(gaze_root, filename_wo_ext + ".npy")
            if os.path.exists(cand_gaze):
                gaze_path = cand_gaze

        if ref_path is None or aud_path is None:
            print(f"[跳过] {subdir}：未找到图像或音频文件")
            continue

        # 生成输出路径
        out_video_path = os.path.join(opt.res_dir, subdir + '.mp4')
        os.makedirs(os.path.dirname(out_video_path), exist_ok=True)

        # 推理
        print(f"[处理] {subdir}")
        agent.run_inference(
            out_video_path,
            ref_path,
            aud_path,
            pose_path,   # 允许 None
            gaze_path,   # 允许 None
            a_cfg_scale=opt.a_cfg_scale,
            nfe=opt.nfe,
            no_crop=opt.no_crop,
            seed=opt.seed
        )

    print("✅ 批量推理完成。")


