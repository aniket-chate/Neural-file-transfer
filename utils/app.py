import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from flask import Flask, render_template, request, send_from_directory

from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm

from werkzeug.utils import secure_filename

from wtforms import (
    FileField,
    SubmitField,
    FloatField,
    HiddenField
)

from models import VGGEncoder, Decoder
from utils import adaptive_instance_normalization


# ============================================================
# Project Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
EXAMPLES_DIR = BASE_DIR / "examples"

VGG_PATH = BASE_DIR / "utils" / "vgg_normalised.pth"
DECODER_PATH = BASE_DIR / "experiment" / "final_exp" / "decoder_final.pth"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Flask App
# ============================================================

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR)
)

app.config["SECRET_KEY"] = "supersecretkey"

app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

app.config["ALLOWED_EXTENSIONS"] = {
    "jpg",
    "jpeg",
    "png"
}

Bootstrap(app)


# ============================================================
# Upload Form
# ============================================================

class UploadForm(FlaskForm):

    content = FileField("Content Image")

    style = FileField("Style Image")

    content_path = HiddenField()

    style_path = HiddenField()

    alpha = FloatField(
        "Alpha",
        default=1.0
    )

    submit = SubmitField("Transfer Style")


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device : {device}")


# ============================================================
# Load Encoder
# ============================================================

encoder = VGGEncoder(str(VGG_PATH)).to(device)

encoder.eval()


# ============================================================
# Load Decoder
# ============================================================

decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(
        DECODER_PATH,
        map_location=device
    )
)

decoder.eval()
# ============================================================
# Utility Functions
# ============================================================

def allowed_file(filename):
    """
    Check whether uploaded file has an allowed extension.
    """
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


# ============================================================
# Style Transfer
# ============================================================

def style_transfer(content_image, style_image, encoder, decoder, alpha, device):

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    content_tensor = transform(content_image).unsqueeze(0).to(device)
    style_tensor = transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():

        content_feature = encoder(
            content_tensor,
            is_test=True
        )

        style_feature = encoder(
            style_tensor,
            is_test=True
        )

        target_feature = adaptive_instance_normalization(
            content_feature,
            style_feature
        )

        target_feature = (
            alpha * target_feature +
            (1 - alpha) * content_feature
        )

        output = decoder(target_feature)

    return output


# ============================================================
# Save Tensor Image
# ============================================================

def save_image(image_tensor, save_path):

    image = image_tensor.detach().cpu().squeeze(0)

    image = image.clamp(0, 1)

    image = transforms.ToPILImage()(image)

    image.save(save_path)


# ============================================================
# Load Image
# ============================================================

def load_image(path):

    with Image.open(path) as img:
        return img.convert("RGB")
    
# ============================================================
# Main Route
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if request.method == "POST":

        try:

            alpha = float(form.alpha.data or 1.0)

            # -------------------------
            # Content Image
            # -------------------------

            if form.content.data and form.content.data.filename != "":

                if not allowed_file(form.content.data.filename):
                    raise Exception("Unsupported content image format.")

                content_filename = secure_filename(
                    os.path.basename(form.content.data.filename)
                )

                content_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    content_filename
                )

                form.content.data.save(content_path)

                form.content_path.data = content_filename

            else:

                content_filename = form.content_path.data

                if not content_filename:
                    raise Exception("Please upload a content image.")

                content_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    content_filename
                )

            # -------------------------
            # Style Image
            # -------------------------

            if form.style.data and form.style.data.filename != "":

                if not allowed_file(form.style.data.filename):
                    raise Exception("Unsupported style image format.")

                style_filename = secure_filename(
                    os.path.basename(form.style.data.filename)
                )

                style_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    style_filename
                )

                form.style.data.save(style_path)

                form.style_path.data = style_filename

            else:

                style_filename = form.style_path.data

                if not style_filename:
                    raise Exception("Please upload a style image.")

                style_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    style_filename
                )

            # -------------------------
            # Load Images
            # -------------------------

            content_image = load_image(content_path)

            style_image = load_image(style_path)

            # -------------------------
            # Style Transfer
            # -------------------------

            output = style_transfer(
                content_image,
                style_image,
                encoder,
                decoder,
                alpha,
                device
            )

            # -------------------------
            # Save Result
            # -------------------------

            result_filename = f"stylized_{content_filename}"

            result_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                result_filename
            )

            save_image(output, result_path)

            result_image = result_filename

        except Exception as e:

            error = str(e)

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


# ============================================================
# Uploaded Images
# ============================================================

@app.route("/uploads/<path:filename>")
def send_image(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ============================================================
# Example Images
# ============================================================

@app.route("/examples/<path:filename>")
def send_example(filename):

    return send_from_directory(
        EXAMPLES_DIR,
        filename
    )


# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":

    print("\n==============================")
    print("AdaIN Style Transfer")
    print("==============================")
    print(f"Device : {device}")
    print(f"Templates : {TEMPLATE_DIR}")
    print(f"Static : {STATIC_DIR}")
    print(f"Uploads : {UPLOAD_DIR}")
    print("==============================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )