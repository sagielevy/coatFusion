# CoatFusion

This repository contains the official codebase for the CoatFusion paper. It includes the scripts for dataset generation (via Blender) and the model training/inference pipeline (via Hugging Face Diffusers).

## Repository Structure

- `coating_dataset_generator/`: Blender Python (`bpy`) scripts for procedurally generating the material coating dataset. 
- `material_coating_training/`: PyTorch & Diffusers scripts for training a DreamBooth LoRA for the inpainting task, and running inference.

## Downloading Assets & Datasets

To keep this repository lightweight, all heavy assets and datasets are hosted on Hugging Face at [ANONYMIZED_LINK](#).

This includes:
- **Training dataset:** `coating_dataset_Training/`
- **Benchmarking dataset:** `coating_dataset_Benchmark/`
- **Generator assets:** `Models/`, `HDRIs/`, Blender files, etc.
- **Pretrained Weights:** Our trained LoRA weights and material embeddings are available in [`model_weights/checkpoint-35000`](#). You can see an example of how to use them for inference in `material_coating_training/inference_flux_fill_interactive.ipynb`.

### Automatic Download (Recommended)
You can easily download the required assets directly into the correct folders using the `huggingface_hub` CLI. 

1. Ensure you have the library installed:
```bash
pip install huggingface_hub
```

2. Run the following command from the root of this repository to download everything into the `coating_dataset_generator` directory:
```bash
huggingface-cli download <ANONYMIZED_LINK> --local-dir coating_dataset_generator --repo-type dataset
```

### Manual Download
If you prefer, you can visit [ANONYMIZED_LINK](#) and manually download the `.blend` file or specific directories (`Models/`, `HDRIs/`, etc.) and place them inside the `coating_dataset_generator/` directory.

## Setup Instructions

We recommend creating separate virtual environments for the dataset generator and the training pipeline. We have provided `requirements.txt` files in each respective directory.

**Dataset Generator Setup:**
```bash
cd coating_dataset_generator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Training & Inference Setup:**
```bash
cd material_coating_training
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
