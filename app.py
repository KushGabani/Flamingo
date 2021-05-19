import io
import warnings
from flask import Flask, url_for, render_template

warnings.filterwarnings("ignore")

app = Flask(__name__)


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


if __name__ == "__main__":
    app.run(debug=True)
