<div align="center">
<p align="center">
  <h1>IMTalker: Efficient Audio-driven Talking Face Generation with Implicit Motion Transfer</h1>

  [![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/000000)
  [![Hugging Face Model](https://img.shields.io/badge/Model-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/cbsjtu01/IMTalker)
  [![Hugging Face Space](https://img.shields.io/badge/Space-HuggingFace-blueviolet?logo=huggingface)](https://huggingface.co/spaces/chenxie95/IMTalker)
  [![demo](https://img.shields.io/badge/GitHub-Demo-orange.svg)](https://cbsjtu01.github.io/IMTalker/)


</p>
</div>

## 📖 Overview
IMTalker accepts diverse portrait styles and achieves 40 FPS for video-driven and 42 FPS for audio-driven talking-face generation when tested on an NVIDIA RTX 4090 GPU at 512 × 512 resolution. It also enables diverse controllability by allowing precise head-pose and eye-gaze inputs alongside audio

<div align="center">
  <img src="assets/teaser.png" alt="" width="1000">
</div>

## 📢 News
- **[2025.11]** 🚀 The inference code and pretrained weights are released!
## 🛠️ Installation

### 1. Environment Setup

```bash
conda create -n IMTalker python=3.10
conda activate IMTalker
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

**2. Install with pip:**

```bash
git clone https://github.com/cbsjtu01/IMTalker.git
cd IMTalker
pip install -r requirement.txt
```
## ⚡ Quick Start

You can simply run the Gradio demo to get started. The script will **automatically download** the required pretrained models to the `./checkpoints` directory if they are missing.

```bash
python app.py
```

## 📦 Model Zoo

Please download the pretrained models and place them in the `./checkpoints` directory.

| Component | Checkpoint | Description | Download |
| :--- | :--- | :--- | :---: |
| **Audio Encoder** | `wav2vec2-base-960h` | Wav2Vec2 Base model | [🤗 Link](https://huggingface.co/cbsjtu01/IMTalker/tree/main/wav2vec2-base-960h) |
| **Generator** | `generator.ckpt` | Flow Matching Generator | [🤗 Link](https://huggingface.co/cbsjtu01/IMTalker/blob/main/generator.ckpt) |
| **Renderer** | `renderer.ckpt` | IMT Renderer | [🤗 Link](https://huggingface.co/cbsjtu01/IMTalker/blob/main/renderer.ckpt) |
### 📂 Directory Structure
Ensure your file structure looks like this after downloading:

```text
./checkpoints
├── renderer.ckpt                     # The main renderer
├── generator.ckpt                    # The main generator
└── wav2vec2-base-960h/               # Audio encoder folder
    ├── config.json
    ├── pytorch_model.bin
    └── ...
```

## 🚀 Inference

### 1. Audio-driven Inference
Generate a talking face from a source image and an audio file.

```bash
python generator/generate.py \
    --ref_path "./assets/source_image.jpg" \
    --aud_path "./assets/input_audio.wav" \
    --res_dir "./results/" \
    --generator_path "./checkpoints/generator.ckpt" \
    --renderer_path "./checkpoints/renderer.ckpt" \
    --a_cfg_scale 3 \
    --crop
```
### 2. Video-driven Inference
Generate a talking face from a source image and an driving video file.

```bash
python renderer/inference.py \
    --source_path "./assets/source_image.jpg" \
    --driving_path "./assets/driving_video.mp4" \
    --save_path "./results/" \
    --renderer_path "./checkpoints/renderer.ckpt" \
    --crop
```

## 💡 Best Practices

To obtain the highest quality generation results, we recommend following these guidelines:

1.  **Input Image Composition**: 
    Please ensure the input image features the person's head as the primary subject. Since our model is explicitly trained on facial data, it does not support full-body video generation. 
    * The inference pipeline automatically **crops the input image** to focus on the face by default.
    * **Note on Resolution**: The model generates video at a fixed resolution of **512×512**. Using extremely high-resolution inputs will result in downscaling, so prioritize facial clarity over raw image dimensions.

2.  **Audio Selection**: 
    Our model was trained primarily on **English datasets**. Consequently, we recommend using **English audio** inputs to achieve the best lip-synchronization performance and naturalness.

3.  **Background Quality**: 
    We strongly recommend using source images with **solid colored** or **blurred (bokeh)** backgrounds. Complex or highly detailed backgrounds may lead to visual artifacts or jitter in the generated video.

## 📝 To-Do List
- [x] Release inference code and pretrained models.
- [x] Launch Hugging Face online demo.
- [ ] Release training code and data preprocessing scripts.

## 📜 Citation
If you find our work useful for your research, please consider citing:

```bibtex
@article{imtalker2025,
  title={IMTalker: Efficient Audio-driven Talking Face Generation with Implicit Motion Transfer},
  author={Bo, Chen and Tao, Liu and Qi, Chen and  Xie, Chen and  Zilong Zheng}, 
  journal={arXiv preprint arXiv:2511.22167},
  year={2025}
}
```

## 🙏 Acknowledgement

We express our sincerest gratitude to the excellent previous works that inspired this project:

- **[IMF](https://github.com/ueoo/IMF)**: We adapted the framework and training pipeline from IMF and its reproduction code.
- **[FLOAT](https://github.com/deepbrainai-research/float)**: We referenced the model architecture and implementation of Float for our generator.
- **[Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h)**: We utilized Wav2Vec as our audio encoder.
- **[Face-Alignment](https://github.com/1adrianb/face-alignment)**: We used FaceAlignment for cropping images and videos.