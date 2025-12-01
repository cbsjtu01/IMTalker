import argparse
import os
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import numpy as np
import face_alignment
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from renderer.models import IMTRenderer

# Optimize CUDA backends
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class DataProcessor:
    def __init__(self, opt):
        self.opt = opt
        self.input_size = opt.input_size
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
        ])

    @torch.no_grad()
    def process_img(self, img):
        """Detects face and crops the image to the face region."""
        if isinstance(img, Image.Image):
            img = np.array(img)
            
        # Ensure RGB
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
        # Resize for faster detection
        h, w = img.shape[:2]


        bboxes = self.fa.face_detector.detect_from_image(img)
        
        # Filter valid faces (score > 0.95)
        valid_bboxes = [
            (int(x1), int(y1), int(x2 ), int(y2 ), score)
            for (x1, y1, x2, y2, score) in bboxes if score > 0.95
        ]

        if not valid_bboxes:
            print("[WARN] No face detected, falling back to center resize.")
            return cv2.resize(img, (self.input_size, self.input_size))
    
        # Crop logic based on the first detected face
        x1, y1, x2, y2, _ = valid_bboxes[0]
        bsy, bsx = int((y2 - y1) / 2), int((x2 - x1) / 2)
        my, mx = int((y1 + y2) / 2), int((x1 + x2) / 2)
        bs = int(max(bsy, bsx) * 1.3)
    
        # Pad image to allow cropping outside boundaries
        img = cv2.copyMakeBorder(img, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
        
        # Adjust coordinates for padding
        my, mx = my + bs, mx + bs
        crop_img = img[my - bs:my + bs, mx - bs:mx + bs]
        return Image.fromarray(crop_img)

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)


def save_video(vid_target_recon, save_path, fps):
    vid = vid_target_recon.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()
    vid = (vid * 255).astype(np.uint8)
    T, H, W, C = vid.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

    for frame in vid:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"[Success] Video saved to {save_path}")


class Demo(nn.Module):
    def __init__(self, args, gen):
        super(Demo, self).__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print('==> Loading model...')
        self.gen = gen.to(self.device)
        self.gen.eval()

        self.save_path = args.save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.processor = DataProcessor(args)

    @torch.no_grad()
    def process_single(self, source_path, driving_path):
        print(f"==> Processing: {source_path} + {driving_path}")
        
        # 1. Process Source Image
        source_img = self.processor.load_image(source_path)
        if self.args.crop:
            source_img = self.processor.process_img(source_img)
        
        source_tensor = self.processor.transform(source_img).unsqueeze(0).to(self.device)

        # 2. Encode Source Appearance & Motion
        f_r, i_r = self.gen.app_encode(source_tensor)
        t_r = self.gen.mot_encode(source_tensor)
        ta_r = self.gen.adapt(t_r, i_r)
        ma_r = self.gen.mot_decode(ta_r)

        # 3. Process Driving Video Frame-by-Frame
        cap = cv2.VideoCapture(driving_path)
        fps = cap.get(cv2.CAP_PROP_FPS) if self.args.fps is None else self.args.fps
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        vid_target_recon = []
        
        pbar = tqdm(total=frame_count, desc="Inferencing")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame)
            
            # Transform driving frame
            frame_tensor = self.processor.transform(frame_pil).unsqueeze(0).to(self.device)

            # Inference
            t_c = self.gen.mot_encode(frame_tensor)
            ta_c = self.gen.adapt(t_c, i_r)
            ma_c = self.gen.mot_decode(ta_c)
            out = self.gen.decode(ma_c, ma_r, f_r)
            
            vid_target_recon.append(out.cpu()) # Move to CPU immediately to save VRAM
            pbar.update(1)
            
        cap.release()
        pbar.close()

        if not vid_target_recon:
            print("[Error] No frames generated.")
            return

        # 4. Save Result
        vid_target_recon = torch.cat(vid_target_recon, dim=0)
        save_name = f"{Path(source_path).stem}_{Path(driving_path).stem}.mp4"
        save_video(vid_target_recon, os.path.join(self.save_path, save_name), fps)

    def process_batch(self, root_dir):
        subdirs = [
            os.path.join(root_dir, d) for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        
        for sub in subdirs:
            img_files = [f for f in os.listdir(sub) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            vid_files = [f for f in os.listdir(sub) if f.lower().endswith((".mp4", ".avi", ".mov"))]
            
            if not img_files or not vid_files:
                print(f"[Skip] {sub} missing image or video.")
                continue
                
            img_path = os.path.join(sub, img_files[0])
            vid_path = os.path.join(sub, vid_files[0])
            
            try:
                self.process_single(img_path, vid_path)
            except Exception as e:
                print(f"[Error] Failed processing {sub}: {e}")

    def run(self):
        if self.args.source_path and self.args.driving_path:
            self.process_single(self.args.source_path, self.args.driving_path)
        elif self.args.data_dir:
            self.process_batch(self.args.data_dir)
        else:
            raise ValueError("Usage: Provide either --source_path & --driving_path OR --data_dir")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Animation Inference Demo")

    # Modes
    parser.add_argument("--source_path", type=str, help="Path to source image")
    parser.add_argument("--driving_path", type=str, help="Path to driving video")
    parser.add_argument("--data_dir", type=str, help="Batch directory containing subfolders")
    parser.add_argument("--save_path", type=str, default="./results", help="Output directory")

    # Model Params
    parser.add_argument("--renderer_path", type=str, default="./checkpoints/renderer.ckpt", help="Checkpoint path")
    parser.add_argument("--input_size", type=int, default=256, help="Resolution")
    parser.add_argument('--swin_res_threshold', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--window_size', type=int, default=8)
    
    # Options
    parser.add_argument("--fps", type=int, default=None, help="Output FPS (default: same as input)")
    parser.add_argument("--crop", action="store_true", help="Crop face from source image")

    args = parser.parse_args()

    # Initialize Model
    model = IMTRenderer(args)
    checkpoint = torch.load(args.renderer_path, map_location="cpu")
    
    # Handle state dict keys
    state_dict = checkpoint.get("state_dict", checkpoint)
    clean_state_dict = {
        k.replace("gen.", ""): v for k, v in state_dict.items() if k.startswith("gen.")
    }
    
    model.load_state_dict(clean_state_dict, strict=False)

    # Run Demo
    demo = Demo(args, model)
    demo.run()
