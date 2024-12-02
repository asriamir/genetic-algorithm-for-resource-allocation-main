### U-Net Image Segmentation
This repository contains an implementation of the U-Net architecture for semantic image segmentation using TensorFlow and Keras. The model is designed for the task of image segmentation and predicts the pixel-wise segmentation mask for input images.

### Overview
The U-Net model is a fully convolutional network designed for semantic segmentation tasks. It is widely used in medical image segmentation, but can be applied to any image segmentation problem. This implementation is customized to work with RGB images and segmentation masks, and it includes data preprocessing, model definition, training, and prediction steps.

### Key Components:
Model Architecture: U-Net is an encoder-decoder architecture that captures both high-level context and fine-grained details for pixel-wise prediction.
Data Preprocessing: The dataset is processed to handle image and mask loading, resizing, and normalization.
Training: The model is trained on the dataset with a custom loss function and evaluation metrics.
Prediction: The trained model is used to predict segmentation masks for new images.
### Requirements
TensorFlow 2.x
NumPy
Imageio
Matplotlib
Pandas
You can install the required dependencies by running the following command:

```bash

pip install tensorflow numpy imageio matplotlib pandas
```
File Structure
```bash
├── data/
│   ├── CameraRGB/      # Folder containing the input RGB images
│   └── CameraMask/     # Folder containing the corresponding segmentation masks
├── test_utils.py       # Utility functions for model summary and comparison
├── main.py             # The main script to run the model
└── README.md           # This file
```
### How to Run
Prepare the Dataset: Ensure that the dataset is stored in the following folder structure:

data/CameraRGB/: Input RGB images
data/CameraMask/: Corresponding segmentation masks
Load and Process Data:

The script automatically loads and processes the images and masks from the dataset.
Images are resized to (96, 128) pixels for consistent input dimensions.
Model Training:

The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss.
Training is conducted for a specified number of epochs (EPOCHS = 40), with batching and shuffling enabled for better generalization.
Model Predictions:

After training, the model can be used to predict segmentation masks for new images.
The predicted masks are displayed alongside the input images and their ground truth masks.
Run the script as follows:

```bash
python main.py
```
### Model Architecture
#### The U-Net architecture used in this implementation consists of:

Contracting Path (Encoder): A series of convolutional blocks followed by max-pooling layers. These layers extract increasingly abstract features from the input image.
Bottleneck: The deepest layer of the network, where the spatial resolution is minimized.
Expansive Path (Decoder): A series of up-sampling layers (using Conv2DTranspose) and concatenation with corresponding contracting path layers. These layers upsample the feature maps to match the original image resolution.
#### U-Net Architecture Summary:
4 levels of convolutional blocks in the encoder.
4 levels of upsampling blocks in the decoder.
The final layer outputs a mask of size (96, 128, 23) representing the predicted segmentation.
#### Example
After training, the model predicts a segmentation mask for an image. The predicted mask, along with the original image and ground truth mask, can be visualized using the following code:

python
```bash
display([sample_image, sample_mask, create_mask(unet.predict(sample_image[tf.newaxis, ...]))])
```
This will show the input image, its true segmentation mask, and the predicted mask by the trained model.

#### Training Hyperparameters
EPOCHS: Number of training epochs. Default is 40.
BATCH_SIZE: Batch size for training. Default is 32.
BUFFER_SIZE: Shuffle buffer size. Default is 500.
Validation Subsplits: Number of validation splits (for cross-validation). Default is 5.
#### License
This project is licensed under the MIT License.