import os
import numpy as np
from PIL import Image

class KNNImageClassifier:
    def __init__(self, training_dir):
        self.training_dir = training_dir
        self.training_data = []

    # Load the training images and their labels
    def load_training_data(self):
        for root, _, files in os.walk(self.training_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(root, file)
                    label = os.path.basename(root)
                    self.training_data.append((image_path, label))

    def train_classifier(self):
        # Load training data
        self.load_training_data()

    def load_image(self, image_path):
        image = Image.open(image_path).convert("L")
        image_data = np.array(image).flatten()
        return image_data

    def classify_image(self, test_image_path):
        if not self.training_data:
            raise ValueError("Training data has not been loaded. Call 'load_training_data()' first.")

        test_image = self.load_image(test_image_path)

        results = []

        # Compare the test image to training images and find the closest ones
        for training_image_path, label in self.training_data:
            training_image = self.load_image(training_image_path)

            # Resize the test image to match the dimensions of the training image
            test_image_resized = np.resize(test_image, training_image.shape)

            distance = np.linalg.norm(training_image - test_image_resized)
            results.append((training_image_path, distance, label))

        # Sort the results by distance and find the most common label among the closest images
        results.sort(key=lambda x: x[1])

        k_nearest = results[:5]  # Assuming a default of 5 neighbors
        class_votes = {}

        for _, _, label in k_nearest:
            class_votes[label] = class_votes.get(label, 0) + 1

        predicted_class = max(class_votes, key=class_votes.get)
        return predicted_class
