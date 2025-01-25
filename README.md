# Flower_Image Classification Using Transfer Learning

This project demonstrates how to classify flower images using **Transfer Learning** and **Data Augmentation** with TensorFlow and Keras. The model is built on the MobileNet architecture and fine-tuned for a custom flower dataset.

## Features
- Uses **MobileNet** as a pre-trained model for feature extraction.
- Implements **Data Augmentation** to improve model generalization.
- Classifies flower images into categories like daisy, dandelion, roses, sunflowers, and tulips.

## Requirements
- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ruchi-lab/Image-classification_flowers.git
   cd flower-classification

2. Install dependencies:
    ```bash
    pip install tensorflow numpy matplotlib

## Usage
Organize your dataset into subfolders (e.g., daisy/, roses/).
```
flowers/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
├── class2/
│   ├── image3.jpg
│   ├── image4.jpg
└── ...
```

### Update the paths in the script for your dataset and test image.

## Run the script:
    python flower_dataset.py
    
## Results
1. The model predicts the class of a test image (e.g., "roses").
2. Training and validation accuracy/loss are visualized using Matplotlib.

## License
This project is open-source and available under the MIT License.
