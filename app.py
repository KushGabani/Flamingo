import os
import warnings
from flask_dropzone import Dropzone
from flask import Flask, url_for, render_template, request
from music_generation import utils as music_utils
from music_generation.model import MusicGenerationModel
from style_transfer import utils as nst_utils
from style_transfer.model import TransformerNet
warnings.filterwarnings("ignore")

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILES=1,
)

dropzone = Dropzone(app)


def generate_music(num_notes, artist):
    print("-----------------process: generating music")
    x, pitchnames, n_vocab = music_utils.get_data(artist)
    mgModel = MusicGenerationModel(artist, n_vocab, (x.shape[1], x.shape[2]))
    mgModel.init_model_architecture()
    model = mgModel.load_model_weights()
    prediction_output = music_utils.generate_sequence(
        model, num_notes, x, pitchnames, n_vocab)
    music = music_utils.generate_notes(prediction_output)
    filename = music_utils.save_midi(music, artist)
    print("-----------------process: end")

    return filename


def transfer_style(content_image, style_image):
    print("-----------------process: getting images")
    nstModel = TransformerNet()
    output_filepath = nst_utils.neural_style_transfer(
        nstModel, content_image, style_image)
    print("-----------------process: end")
    return output_filepath


@app.route("/", methods=['GET'])
def home():
    return render_template("home.html")


@app.route("/about", methods=['GET'])
def about():
    return render_template("about.html")


@app.route("/music-generation", methods=['GET'])
def music_generation():
    return render_template("music_generation.html")


@app.route("/style-transfer", methods=['GET'])
def style_transfer():
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


@app.route("/handle-music-form", methods=['GET', 'POST'])
def success1():
    if request.method == 'POST':
        try:
            duration = int(request.form.get("duration"))
            artist = request.form.get("artist")
            num_notes = int((duration * 150) / 38)
            print(f"duration: {duration}, type: {type(duration)}")
            print(f"artist: {artist}, type: {type(artist)}")
            print(f"num_notes: {num_notes}, type: {type(num_notes)}")
            filename = generate_music(num_notes, artist)
            return render_template("success.html", artist=artist, audio_src=filename, is_music_module="true")
        except:
            return "<div style='width: 100%; height: 100%; display:flex; justify-content:center; align-items: center'><h1 style='color:#ff4242;'>You didn't provide the needed information</h1></div>"
    else:
        return render_template("success.html", artist="Schubert", audio_src='created_music/schubert.mid', is_music_module="true")


@app.route("/handle-transfer-form", methods=['GET', 'POST'])
def success2():
    if request.method == 'POST':
        # try:
        style_filepath = f"static/style_images/{request.form.get('style-image')}.jpg"
        filename = transfer_style("uploads/content.jpg", style_filepath)
        return render_template("success.html", image_src=filename, is_music_module="false")
        # except:
        #     return "<div style='width: 100%; height: 100%; display:flex; justify-content:center; align-items: center'><h1 style='color:#ff4242;'>You didn't provide the needed information</h1></div>"
    else:
        return render_template("success.html", image_src='style_transfer/output.png', is_music_module="false")


if __name__ == "__main__":
    app.run(debug=True)
