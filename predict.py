import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("model/brain_tumor_model.h5")

# Class names (must match your dataset folders)
class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    return class_names[class_index]