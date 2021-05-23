import io
import os
import warnings
from flask_dropzone import Dropzone
from flask import Flask, url_for, render_template, request
warnings.filterwarnings("ignore")

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILES=1,
    # DROPZONE_UPLOAD_ON_CLICK=True
)

dropzone = Dropzone(app)


@app.route("/", methods=['GET'])
def home():
    return render_template("home.html")


@app.route("/about", methods=['GET'])
def about():
    return render_template("about.html")


@app.route("/music-generation", methods=['GET'])
def music_generation():
    return render_template("music_generation.html")


@app.route("/style-transfer", methods=['GET', 'POST'])
def style_transfer():
    if request.method == 'POST':
        print(request.form.get("style-image"))
    return render_template("style_transfer.html")

@app.route("/upload-content", methods=['POST'])
def upload_content_image():
    print(request.files)
    for key, f in request.files.items():
        if key.startswith('file'):
            f.save(os.path.join(app.config['UPLOADED_PATH'], "content.jpg"))
    return ""

@app.route("/curated-collection", methods=['GET'])
def collection():
    return render_template("curated_collection.html")

if __name__ == "__main__":
    app.run(debug=True)
