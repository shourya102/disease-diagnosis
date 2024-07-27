import numpy as np
from keras.models import load_model
from keras.preprocessing.image import image_utils


def predict_lungs(img_path):
    model = load_model('lungs.h5')
    test_image = image_utils.load_img(img_path, target_size=(224, 224))
    test_image = image_utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image, batch_size=1)
    print(result)
    if result[0][1] == 0:
        return "Bacterial Pneumonia"
    elif result[0][1] == 1:
        return "Corona Virus Disease"
    elif result[0][1] == 2:
        return "Normal"
    elif result[0][1] == 3:
        return "Tuberculosis"
    else:
        return "Viral Pneumonia"


def predict_heart(img_path):
    model = load_model('heart.h5')
    test_image = image_utils.load_img(img_path, target_size=(224, 224))
    test_image = image_utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image, batch_size=1)
    print(result)
    if result[0][1] == 0:
        return "Normal"
    else:
        return "Sick"


def predict_brain(img_path):
    model = load_model('brain.h5')
    test_image = image_utils.load_img(img_path, target_size=(224, 224))
    test_image = image_utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image, batch_size=1)
    print(result)
    if result[0][1] == 0:
        return "Glioma"
    elif result[0][1] == 1:
        return "Meningioma"
    elif result[0][1] == 2:
        return "No Tumor"
    else:
        return "Pituitary"
