import io
import os
import warnings
from flask_dropzone import Dropzone
from flask import Flask, url_for, render_template, request
# from flask_uploads import UploadSet, configure_uploads, IMAGES
warnings.filterwarnings("ignore")

app = Flask(__name__)
dropzone = Dropzone(app)

# app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
# app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
# app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
# app.config['DROPZONE_REDIRECT_VIEW'] = 'style_transfer'
# app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# app.config['DROPZONE_MAX_FILES'] = 2

# photos = UploadSet('photos', IMAGES)
# configure_uploads(app, photos)

app.config.update(
    UPLOADED_PATH=os.path.join(os.getcwd(), 'uploads'),
    DROPZONE_MAX_FILE_SIZE=1024,
    DROPZONE_TIMEOUT=5*60*1000,
    DROPZONE_MAX_FILES=2
)


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
        f = request.files.get('file')
        f.save(os.path.join(app.config['UPLOADED_PATH'], f.filename))
    return render_template("style_transfer.html")
    # file_urls = []
    # print(request)
    # if request.method == 'POST':
    #     file_obj = request.files
    #     for f in file_obj:
    #         file = request.files.get(f)
    #         print(file)
    #         filename = photos.save(
    #             file,
    #             name=file.filename
    #         )
    #         # append image urls
    #         file_urls.append(photos.url(filename))
    #     print(file_urls)
    # return "uploading..."


@ app.route('/deletefile', methods=['POST'])
def delete_file():
    if request.method == 'POST':
        print("--------------------------------HERE------------------------------")
        filename = request.form['name']
        file_path = os.path.join(os.getcwd() + '/uploads', filename)
        os.remove(file_path)
        return "file deleted"


if __name__ == "__main__":
    app.run(debug=True)
