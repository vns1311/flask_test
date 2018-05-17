from base64 import b64decode
from os import listdir, getcwd
from os.path import isfile, join
from time import strftime

from flask import Flask, request, render_template

from util import check_for_face

app_dir = getcwd()
PATH_TO_IMAGE_DIR = app_dir + '/train_images'
HAAR_CASCADE_PATH = app_dir + '/static/haarcascade_frontalface_default.xml'
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        img_data = request.form['imgData']
        user_id = request.form['userID']
        img = b64decode(img_data)
        list_users = [file for file in listdir(PATH_TO_IMAGE_DIR) if isfile(join(PATH_TO_IMAGE_DIR, file))]
        images_count = len([image_name for image_name in list_users if user_id.lower() in image_name.lower()])
        number_of_faces = check_for_face(img, HAAR_CASCADE_PATH, app)
        if number_of_faces == 1:
            filename = PATH_TO_IMAGE_DIR + '/%s_%s.png' % (user_id, strftime("%Y%m%d_%H%M%S"))
            with open(filename, 'wb') as img_file:
                img_file.write(img)
            list_users = [file for file in listdir(PATH_TO_IMAGE_DIR) if isfile(join(PATH_TO_IMAGE_DIR, file))]
            images_count = len([image_name for image_name in list_users if user_id.lower() in image_name.lower()])
            app.logger.info("%s images captured", str(images_count))
        return (str(images_count), 200)


if __name__ == '__main__':
    app.run(debug=True)
