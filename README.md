# Image Display Application

This Python application allows users to open images, classify them using a KNN (K-Nearest Neighbors) image classifier, and save the image data along with their classifications.

## Requirements
- Python 3.x
- PyQt6
- NumPy
- PIL (Python Imaging Library)

## Usage
1. Run the application by executing the script `image_display.py`.
2. Either provide the training directory path as a command-line argument or enter it when prompted.
3. Click on the "Open Image" button to select an image file (PNG, JPG, JPEG, BMP, GIF).
4. The application will display the selected image and predict its class using the KNN image classifier.
5. Optionally, click on the "Save Data" button to save the image data along with its classification to a JSON file.
6. Close the application by clicking on the "Close" button or closing the window.

## Features
- **Open Image**: Select an image file from your system.
- **Display Image**: View the selected image within the application.
- **Classify Image**: Predict the class of the image using a pre-trained KNN image classifier.
- **Save Data**: Save the image data and its classification to a JSON file.
- **Close**: Close the application.

## Application Structure
- **ImageDisplayApp**: Main window of the application containing buttons and labels for image display and classification.
- **KNNImageClassifier**: Class responsible for training and classifying images using the KNN algorithm.
- **openFileNameDialog**: Method to open a file dialog and select an image file.
- **close_app**: Method to close the application and print image data and classifications.
- **save_data**: Method to save image data and classifications to a JSON file.
- **main**: Entry point of the application, initializes the GUI and starts the event loop.

## Note
- Ensure that the required Python packages are installed before running the application.
- The KNN classifier requires a training dataset stored in the specified directory.
- This application provides a simple interface for image classification and data saving, suitable for educational or experimental purposes.

For any questions or feedback, please contact the developer at aghilesasmani@gmail.com

**Developer:** Ghiles Asmani
