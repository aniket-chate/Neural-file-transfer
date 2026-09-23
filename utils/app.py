import base64
import io
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

from utils.models import VGGEncoder, Decoder
from utils.utils import (
    adaptive_instance_normalization,
    calc_mean_std
)


# ============================================================
# Project Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = Path("/tmp/neuralart_uploads")
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

    alpha = HiddenField(
        "Alpha",
        default="1.0"
    )

    submit = SubmitField("Transfer Style")


# ============================================================
# Device
# ============================================================

device = torch.device("cpu")
torch.set_num_threads(1)
print(f"Using device : {device}")


encoder = None
decoder = None

def load_models():
    global encoder, decoder
    
    
    if encoder is None:
        encoder = VGGEncoder(VGG_PATH).to(device)
        encoder.eval()

    if decoder is None:
        decoder = Decoder().to(device)
        decoder.load_state_dict(
            torch.load(
                DECODER_PATH,
                map_location="cpu"
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
        transforms.Resize((256,256)),
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

def image_to_data_uri(image_tensor):
    """Convert the generated tensor directly to an in-memory JPEG data URI.

    Vercel Functions have a read-only deployment filesystem, so generated
    request-specific files must not be written under the project directory.
    """
    image = image_tensor.detach().cpu().squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image).convert("RGB")

    buffer = io.BytesIO()
    image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    image.save(buffer, format="JPEG", quality=78, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


# ============================================================
# Load Image
# ============================================================

def load_image(file_storage):
    """Load an uploaded image directly from the request into memory."""
    file_storage.stream.seek(0)
    with Image.open(file_storage.stream) as img:
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
            alpha = float(request.form.get("alpha", "1.0") or "1.0")
            if not 0.0 <= alpha <= 1.0:
                raise Exception("Style strength must be between 0 and 1.")

            # Uploads are processed directly from request memory.
            # Vercel Functions have a read-only deployment filesystem.
            if not form.content.data or not form.content.data.filename:
                raise Exception("Please upload a content image.")
            if not allowed_file(form.content.data.filename):
                raise Exception("Unsupported content image format. Use JPG, JPEG, or PNG.")

            if not form.style.data or not form.style.data.filename:
                raise Exception("Please upload a style image.")
            if not allowed_file(form.style.data.filename):
                raise Exception("Unsupported style image format. Use JPG, JPEG, or PNG.")

            content_filename = secure_filename(
                os.path.basename(form.content.data.filename)
            ) or "content.jpg"
            style_filename = secure_filename(
                os.path.basename(form.style.data.filename)
            ) or "style.jpg"

            content_image = load_image(form.content.data)
            style_image = load_image(form.style.data)

            # Keep previews in the same response; do not depend on generated
            # files surviving across serverless invocations.
            def preview_data_uri(image):
                buffer = io.BytesIO()
                image.thumbnail((480, 480), Image.Resampling.LANCZOS)
                image.save(buffer, format="JPEG", quality=65, optimize=True)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                return f"data:image/jpeg;base64,{encoded}"

            content_preview = preview_data_uri(content_image)
            style_preview = preview_data_uri(style_image)

            load_models()

            with torch.inference_mode():
                output = style_transfer(
                    content_image,
                    style_image,
                    encoder,
                    decoder,
                    alpha,
                    device
                )

            # Return the generated image in the same HTTP response.
            result_image = image_to_data_uri(output)

        except Exception as e:
            error = str(e)
            content_preview = None
            style_preview = None

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        content_preview=locals().get("content_preview"),
        style_preview=locals().get("style_preview"),
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
    print("==============================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )