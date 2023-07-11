import os
import cv2
import joblib
import numpy as np
from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename

import module as md

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/img/'


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


class_names = ["Greening", "Blackspot", "Canker", "Fresh"]
knn = joblib.load('knn_model.pkl')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=["GET", 'POST'])
def predict():
    if request.method == "POST":
        image = request.files['image']
        if image and allowed_file(image.filename):
            image.save(os.path.join(
                app.config['UPLOAD_FOLDER'], secure_filename(image.filename)))
            image_path = os.path.join(
                app.config['UPLOAD_FOLDER'], secure_filename(image.filename))
            image_cv = cv2.imread(image_path)
            image_cv = cv2.resize(image_cv, (256, 256))
            features = md.extract_features(image_cv)
            prediction = knn.predict([features])
            predicted_class = prediction[0].item()
            result = class_names[predicted_class]

            # Add the description, prevention, and treatment based on the predicted class
            description = ""
            prevention = ""
            treatment = ""

            if predicted_class == 0:  # Greening
                description = "Penyakit Citrus Greening (CVPD) merupakan penyakit yang disebabkan oleh bakteri gram negatif (Liberibacter asiaticus). Penyakit ini dapat menyerang bagian daun dan buah. Gejala khas CVPD yaitu tidak merata perkembangan tanaman terutama pertumbuhan baru yang dapat mengakibatkan belang-belang kuning. Gejala lainnya yang dimiliki yaitu buah menjadi kecil dan mengandung biji yang gugur dan hanya matang di satu sisi. Buah jeruk yang terkena citrus greening ditandai dengan rasa pahit, ukuran mengecil dan mengandung biji kecil berwarna kecoklatan / hitam"
                prevention = "CVPD dapat dikendalikan dengan menggunakan tanaman yang sehat dan bebas CVPD. Selain itu, lokasi kebun setidanya berjarak 5 km dari kebun jeruk yang diserang CVPD."
                treatment = "Sayangnya, tidak ada pengobatan yang efektif untuk Greening. Tindakan yang dapat dilakukan adalah memantau dan mengendalikan serangga vektor penyakit serta merawat tanaman dengan baik agar tetap sehat."

            elif predicted_class == 1:  # Blackspot
                description = "Blackspot merupakan penyakit pada buah jeruk yang disebabkan karna jamur phyllosticta citricarpa. Penyakit tersebut merupakan penyakit dengan gejala memiliki luka-luka bopeng keras seperti bercak yang menyerupai kawah dengan bagian yang terang, tepian coklat-gelap hingga hitam yang dapat menyebar tidak beraturan dalam area yang luas. Bercak-bercak berwarna oranye hingga merah yang dapat berubah menjadi coklat"
                prevention = "1. Memilih bibit tanaman yang sehat/n2. Menanam varietas yang tahan penyakit/n3. Memaksimalkan aliran udara di kebun yang dapat mengurangi kebasahan daun/n4.	Memantau kebun secara rutin/n5.	Melakukan pemupukan dengan benar yang bertujuan untuk meningkatkan daya tahan alami tanaman\n6.	Buang sisa-sisa tanaman yang terinfeksi dan musnahkan\n7.	Jaga agar buah dalam keadaan normal, tidak dingin dan tidak kering untuk menghambat perkembangan luka selama penyimpanan."
                treatment = "Pengobatan Blackspot melibatkan penggunaan fungisida yang efektif dan penerapan tindakan pengendalian penyakit yang tepat waktu. Penting untuk memantau dan merawat tanaman secara teratur."

            elif predicted_class == 2:  # Canker
                description = "Kanker jeruk (citrus canker) merupakan penyakit dengan gejala terdapat bercak berwarna coklat yang dikelilingi halo berwarna kuning yang dapat membuat bagian tengah bercak pecah. Bercak tersebut akan berbentuk cekungan seperti kawah di tengahnya dan memanjang hingga kedalaman 1 mm. Penyakit ini disebabkan oleh bakteri Xanthomonas axonupadis pv. citri. Adanya lesi dalam jumlah besar pada buah dapat menyebabkan buah menjadi kecil dan cacat ketika infeksi awal"
                prevention = "Beberapa tindakan pencegahan Canker meliputi pemangkasan dan pembuangan cabang yang terinfeksi, penggunaan fungisida atau antibakteri yang direkomendasikan, serta memastikan sanitasi yang baik di kebun."
                treatment = "Penyakit tersebut dapat dikendalikan dengan menggunakan fungisida berbahan aktif copper dan antibiotic seperti streptomisin dan kloromisetin. Budidaya teknis dikendalikan sehingga spesies yang rentan penyakit tidak ditanam bersama dengan spesies komersial. Penyemprotan harus dilakukan hanya pada musim hujan sebelum infestasi berat."

            elif predicted_class == 3:  # Fresh
                description = "Tanaman Anda sehat dan tidak terkena penyakit."
                prevention = ""
                treatment = ""

            return render_template("prediction.html", result=result, image_path=image_path, description=description, prevention=prevention, treatment=treatment)

        else:
            return render_template("index.html", error="Silahkan upload gambar dengan format JPG")

    else:
        return render_template("index.html")


if __name__ == '__main__':
    app.run()
