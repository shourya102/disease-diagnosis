import os
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from assistant import filter_response
from image_to_diagnosis import predict_brain, predict_heart, predict_lungs
from input_to_diagnosis import input_to_diagnosis

app = Flask(__name__)
CORS(app)
app.debug = True


@app.route('/text_or_document_response', methods=['POST', 'GET'])
def text_or_document_response():
    messages = request.get_json()
    message = {}
    if messages:
        message = filter_response(messages)
    if message and len(message['data']):
        symptoms = {}
        for i in message['data']:
            symptoms[i] = 1
        diagnosis = input_to_diagnosis(symptoms)
        nsymps = ', '.join(message['data'])
        message1 = (f'From your symptoms: {nsymps}, it appears that you might have: {diagnosis}. It is '
                    f'recommended that you see a doctor and obtain a professional diagnosis.')
        return jsonify({'message': message1})
    elif message and len(message['message']):
        msg = message['message']
        return jsonify({'message': msg})
    else:
        return jsonify({'message': 'Error, try again!'})


@app.route('/image_response', methods=['POST', 'GET'])
def image_response():
    type_of = request.form.get('type').lower()
    img = request.files['img']
    img_path = ''
    if img:
        filename = secure_filename(img.filename)
        tempdir = tempfile.gettempdir()
        img_path = os.path.join(tempdir, filename)
        img.save(img_path)
    res = ''
    if type_of:
        if type_of == 'brain':
            res = predict_brain(img_path)
        elif type_of == 'heart':
            res = predict_heart(img_path)
        else:
            res = predict_lungs(img_path)
    return jsonify({'response': res})


if __name__ == '__main__':
    app.run(debug=True)
