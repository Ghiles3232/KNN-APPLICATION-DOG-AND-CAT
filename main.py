import sys
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QFileDialog
from knn_klass import KNNImageClassifier
from PIL import Image
import json

class ImageDisplayApp(QMainWindow):
    def __init__(self, training_dir):
        super().__init__()
        self.image_data = []
        self.classifier = KNNImageClassifier(training_dir)
        self.classifier.train_classifier()
        self.test_file = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Display")
        self.setGeometry(100, 100, 400, 300)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.open_button = QPushButton("Open Image")
        self.save_button = QPushButton("Save Data")
        self.close_button = QPushButton("Close")
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.layout.addWidget(self.open_button)
        self.layout.addWidget(self.save_button)
        self.layout.addWidget(self.close_button)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.result_label)

        self.open_button.clicked.connect(self.openFileNameDialog)
        self.save_button.clicked.connect(self.save_data)
        self.close_button.clicked.connect(self.close_app)

    def openFileNameDialog(self):
        options = QFileDialog.Option.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)

        if file_path:
            self.test_file = file_path
            image = Image.open(file_path).convert("L")
            image_array = np.array(image)
            binary_array = np.where(image_array > 127, 1, 0)
            data = {'image_path': self.test_file, 'image_data': binary_array.tolist(), 'label': None}
            self.image_data.append(data)

            height, width = image_array.shape
            bytes_per_line = width
            q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)
            self.image_label.setFixedSize(300, 200)

            predicted_class = self.classifier.classify_image(self.test_file)
            self.result_label.setText(f"Predicted Class: {predicted_class}")

    def close_app(self):
        for image_info in self.image_data:
            print(f"Image Path: {image_info['image_path']}, Classification: {image_info['label']}")
        self.close()

    def save_data(self):
        if self.test_file:
            image = Image.open(self.test_file).convert("L")
            image_array = np.array(image)
            binary_array = np.where(image_array > 127, 1, 0)
            data = {'image_path': self.test_file, 'image_data': binary_array.tolist(), 'label': None}
            self.image_data.append(data)
            data_filename = "image_data.json"
            with open(data_filename, "w") as f:
                json.dump(self.image_data, f)
            print(f"Data saved to {data_filename}")
        else:
            print("No file has been selected.")

def main():
    training_dir = sys.argv[1] if len(sys.argv) == 2 else input("Enter the training directory path: ")
    app = QApplication(sys.argv)
    window = ImageDisplayApp(training_dir)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
