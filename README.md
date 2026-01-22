# Waste Classification AI System

This project implements a Convolutional Neural Network (CNN) for classifying waste images into six categories: cardboard, glass, metal, paper, plastic, and trash. The system uses transfer learning with MobileNetV2 and provides real-time predictions with confidence scores through a Streamlit web app.

## Features

- **Waste Classification**: Classifies images into 6 waste categories
- **Confidence Scoring**: Provides prediction confidence and handles low-confidence cases
- **Real-time Predictions**: Web interface for uploading and classifying images
- **Data Augmentation**: Enhanced training with various augmentation techniques
- **Model Evaluation**: Test accuracy of approximately 68%

## Dataset

The dataset consists of images organized in train/ and test/ directories, with subdirectories for each class:
- cardboard
- glass
- metal
- paper
- plastic
- trash

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Dragon41249693/waste-classification.git
cd waste-classification
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install tensorflow streamlit pillow pandas numpy
```

## Usage

### Training the Model

Run the training script:
```bash
python train_model.py
```
This will train the model on the training dataset and save it as `waste_classifier_cnn.h5`.

### Testing the Model

Run the testing script:
```bash
python test_model.py
```
This evaluates the model on sample test images.

### Running the Web App

Start the Streamlit app:
```bash
streamlit run app.py
```
Open your browser to the provided URL and upload waste images for classification.

## Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Size**: 224x224 pixels
- **Output**: 6 classes with softmax activation
- **Training**: Transfer learning with fine-tuning
- **Augmentation**: Rotation, zoom, shift, shear, brightness adjustments

## Results

- **Test Accuracy**: 68.01%
- **Classes**: cardboard, glass, metal, paper, plastic, trash
- **Confidence Threshold**: 35% (predictions below this are rejected)

## Contributing

Feel free to contribute by improving the model accuracy, adding more classes, or enhancing the web interface.

## License

This project is open-source. Please check the license file for details.