import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load model
        model = load_model(os.path.join("model", "model.keras"))

        # Load and preprocess image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Predict
        predictions = model.predict(test_image)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Class index to label mapping
        class_labels = [
            "Cardboard",
            "Food Organics",
            "Glass",
            "Metal",
            "Miscellaneous Trash",
            "Paper",
            "Plastic",
            "Textile Trash",
            "Vegetation"
        ]

        # Get label
        prediction = class_labels[predicted_class]
        return [{ "image": prediction }]
