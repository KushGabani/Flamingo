import io
import warnings
from flask import Flask, url_for, render_template

warnings.filterwarnings("ignore")

app = Flask(__name__)


@app.route("/", methods=['GET'])
def home():
    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
