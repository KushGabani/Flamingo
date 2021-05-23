import io
import os
import warnings
from flask_dropzone import Dropzone
from flask import Flask, url_for, render_template, request
from music_generation import utils as music_utils
from music_generation.model import MusicGenerationModel
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


def generate_music(num_notes, artist):
    print("-----------------process: generating music")
    x, pitchnames, n_vocab = music_utils.get_data(artist)
    mgModel = MusicGenerationModel(artist, n_vocab, (x.shape[1], x.shape[2]))
    mgModel.init_model_architecture()
    model = mgModel.load_model_weights()
    prediction_output = music_utils.generate_sequence(model, num_notes, x, pitchnames, n_vocab)
    music = music_utils.generate_notes(prediction_output)
    filename = music_utils.save_midi(music, artist)
    print("-----------------process: end")

    return filename


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

@app.route("/success", methods=['POST'])
def success():
    try:
        duration = int(request.form.get("duration"))
        artist = request.form.get("artist")
        num_notes = int((duration * 150) / 38)
        print(f"duration: {duration}, type: {type(duration)}")
        print(f"artist: {artist}, type: {type(artist)}")
        print(f"num_notes: {num_notes}, type: {type(num_notes)}")
        filename = generate_music(num_notes, artist)
        return render_template("success.html", artist = artist, audio_src = filename)
    except:
        return "<div style='width: 100%; height: 100%; display:flex; justify-content:center; align-items: center'><h1 style='color:#ff4242;'>You didn't provide the needed information</h1></div>"

if __name__ == "__main__":
    app.run(debug=True)
