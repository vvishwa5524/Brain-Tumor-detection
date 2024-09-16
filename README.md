# Brain Tumor Detection with CNN

This project implements a Convolutional Neural Network (CNN) to detect brain tumors in MRI images. It achieves an accuracy of approximately 85% in differentiating between images with and without tumors.

## Data Preprocessing

The preprocessing pipeline consists of several steps to prepare the images for model training:

* **Skull removal:** Removes skull structures to isolate the brain region using Otsu thresholding and connected component analysis.
* **Segmentation:** Identifies the brain region using thresholding and eliminates background noise.
* **Cropping:** Extracts the brain region for further analysis based on its largest connected component.
* **Resizing:** Resizes all images to a uniform size (e.g., 65x65 pixels) for consistent input dimensions.
* **One-Hot Encoding:** Categorizes images as "no tumor" (label 0) and "tumor" (label 1) using One-Hot Encoding for the model.

## Model Architecture

The CNN architecture is designed to extract relevant features from the preprocessed images and classify them accordingly:

* **Sequential Model:** Uses a sequential stack of layers for efficient information processing.
* **Convolutional Layers:** Extract features from the images through convolution operations with multiple filters at different scales.
* **Pooling Layers:** Downsize the feature maps while preserving critical information using Max Pooling.
* **Batch Normalization:** Improves model stability and gradient flow by normalizing activations in each batch.
* **Dropout Layers:** Introduce randomness by dropping a certain percentage of neurons during training, preventing overfitting.
* **Flatten Layer:** Transforms the extracted feature maps into a 1D vector suitable for the final classification layers.
* **Dense Layers:** Perform non-linear transformations on the flattened features using ReLU activation.
* **Softmax Activation:** Outputs the final class probabilities (tumor or no tumor).

## Training and Evaluation

* **Data Splitting:** Splits the dataset into training (80%) and testing (20%) sets for model evaluation.
* **Model Training:** Trains the CNN model with the Adam optimizer using categorical cross-entropy loss function.
* **Model Evaluation:** Evaluates model performance on the unseen testing set using accuracy metric.

## Deployment

* **Model Saving:** Saves the trained model as `model.h5` for future use in prediction and deployment.
* **User Interface (UI):** A basic UI built with Tkinter allows users to select an MRI image for classification.
* **Prediction:** Preprocesses the user-selected image and feeds it to the loaded model.
* **Output:** Displays the predicted class ("no tumor" or "tumor") along with the confidence level.

## Disclaimer

This project is for research purposes only and should not be used for medical diagnosis. Always consult a qualified healthcare professional for any medical concerns.

## Further Development

* **Larger Datasets:** Train the model on a larger and more diverse dataset to enhance accuracy and generalizability.
* **Data Augmentation:**  Employ techniques like random cropping, rotation, and flipping to artificially increase training data diversity.
* **Hyperparameter Tuning:** Optimize the model architecture and training parameters (e.g., learning rate, number of epochs) for improved performance.
* **Interactive UI:** Develop a more user-friendly interface with visualizations and feedback for better interaction.

## Code

The complete code for this project is available in the project repository. It includes functions for data preprocessing, model building, training, evaluation, and prediction with the user interface.

## Project Repository

(This is the project repository itself and datasets are also provided)

## License

(LICENSE @vvishwa5524 vishwanath Tuduru)


This README.md provides a comprehensive overview of the Brain Tumor Detection project, explaining the data preprocessing steps, model architecture, training procedure, deployment aspects, and disclaimer. It also suggests potential improvements and directs users to the project repository for the complete code. Remember to update the placeholder information with specific details regarding your project setup.
