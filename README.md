# Neural Style Transfer

A PyTorch-based **Neural Style Transfer** project that combines content and style images to generate stylized outputs.

> Note: The repository name is **Neural-file-transfer**, while the implemented project is neural style transfer. The documentation uses the project functionality rather than the repository name.

## Overview

The project explores image stylization using a learned decoder and a VGG-based feature representation. The repository contains training experiments, model checkpoints, example images, and a Flask web interface.

## Core Components

- VGG-based feature extraction
- Neural style transfer pipeline
- Learned decoder
- PyTorch training code
- Flask web interface
- Example content/style images
- Saved experiment checkpoints and generated samples

## Repository Structure

```text
Neural-file-transfer/
├── examples/
├── experiment/
│   ├── big_dataset/
│   ├── experiment2/
│   ├── experiment4/
│   ├── experiment5/
│   ├── final_exp/
│   └── trial/
├── static/
│   └── uploads/
├── templates/
├── utils/
│   ├── app.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
├── requirements.txt
└── .gitignore
```

## Technology Stack

- Python
- PyTorch
- Torchvision
- Flask
- Pillow
- Flask-WTF / WTForms
- Gunicorn

## Running the Web Application

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The Flask application entry point is:

```text
utils/app.py
```

Run it with:

```bash
python utils/app.py
```

## Training

Training utilities are located under:

```text
utils/train.py
utils/models.py
utils/utils.py
```

The `experiment/` directory contains experiment configuration files, model checkpoints, and sample outputs from previous runs.

Large datasets and generated upload content are excluded through `.gitignore`.

## Examples

The `examples/` directory contains content/style examples and stylized outputs that demonstrate the image-transformation workflow.

## Author

**Aniket Ganesh Chate**  
B.Tech — Computer Science & Engineering (Data Science)
