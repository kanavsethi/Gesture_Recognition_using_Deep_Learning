# Gesture Recognition using Deep Learning 👋🤖

## Problem Statement 📜

Gesture recognition is a fascinating application of deep learning, and this project focuses on recognizing five distinct hand gestures using a webcam as the input source. The system captures a sequence of images from the webcam, allowing it to accurately identify and classify the user's gestures. This technology can find applications in various domains, including sign language recognition, human-computer interaction, and more.

## Approach 🛠️

To tackle the gesture recognition problem, two deep learning architectures were used for training:

1. **3-D Convolutional Neural Network (3D CNN) Approach**: This approach leverages a 3D CNN model to analyze the features of the image sequences. It can capture motion information effectively and is faster to train than the CNN-RNN approach.

2. **Convolutional Neural Network - Recurrent Neural Network (CNN-RNN) Approach**: This approach combines the power of CNNs for spatial feature extraction with RNNs to handle sequential data. It is suitable for recognizing gestures as a sequence of images.

Both approaches are implemented using TensorFlow Keras.

## Model Architectures 🏗️

### 3D CNN Model

The 3D CNN model has the following architecture:

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 Conv1-1 (Conv3D)            (None, 18, 84, 84, 64)    5248      
                                                                 
 Conv1-2 (BatchNormalization  (None, 18, 84, 84, 64)   256       
 )                                                               
                                                                 
 Conv1-3 (Activation)        (None, 18, 84, 84, 64)    0         
                                                                 
 Conv1-4 (MaxPooling3D)      (None, 9, 42, 84, 64)     0         
                                                                 
 Conv2-1 (Conv3D)            (None, 9, 42, 84, 128)    221312    
                                                                 
 Conv2-2 (BatchNormalization  (None, 9, 42, 84, 128)   512       
 )                                                               
                                                                 
 Conv2-3 (Activation)        (None, 9, 42, 84, 128)    0         
                                                                 
 Conv2-4 (MaxPooling3D)      (None, 4, 21, 42, 128)    0         
                                                                 
 Conv3-1 (Conv3D)            (None, 4, 21, 42, 256)    884992    
                                                                 
 Conv3-2 (BatchNormalization  (None, 4, 21, 42, 256)   1024      
 )                                                               
                                                                 
 Conv3-3 (Activation)        (None, 4, 21, 42, 256)    0         
                                                                 
 Conv3-4 (MaxPooling3D)      (None, 2, 10, 21, 256)    0         
                                                                 
 Conv4-1 (Conv3D)            (None, 2, 10, 21, 256)    1769728   
                                                                 
 Conv4-2 (BatchNormalization  (None, 2, 10, 21, 256)   1024      
 )                                                               
                                                                 
 Conv4-3 (Activation)        (None, 2, 10, 21, 256)    0         
                                                                 
 Conv4-4 (MaxPooling3D)      (None, 1, 5, 10, 256)     0         
                                                                 
 Flat5 (Flatten)             (None, 12800)             0         
                                                                 
 Drop6-1 (Dropout)           (None, 12800)             0         
                                                                 
 Dense6-2 (Dense)            (None, 512)               6554112   
                                                                 
 Drop7-1 (Dropout)           (None, 512)               0         
                                                                 
 Output (Dense)              (None, 5)                 2565      
                                                                 
=================================================================
Total params: 9,440,773
Trainable params: 9,439,365
Non-trainable params: 1,408
_________________________________________________________________
None
```


## File Structure 📂

- `3D CNN Gesture Recognition.ipynb`: Jupyter notebook for training the gesture recognition model using the 3D CNN approach.
- `CNN-RNN Gesture Recognition.ipynb`: Jupyter notebook for training the model using the CNN-RNN approach.
- `Data/train/*`: Training set images.
- `Data/train.csv`: Mapping of training set images to labels.
- `Data/val/*`: Validation set images.
- `Data/val.csv`: Mapping of validation set images to labels.
- `CheckPoints/*`: Model checkpoints saved per epoch during training.

## Getting Started 🚀

1. Clone this repository to your local machine.
2. Install the necessary dependencies, including TensorFlow and other required libraries.
3. Open and run the provided Jupyter notebooks to train and evaluate the gesture recognition models.
4. Experiment, fine-tune the models, and explore different hyperparameters to achieve optimal performance.

## Usage 🤖

Once trained, these models can be used to recognize hand gestures in real-time webcam streams, opening up possibilities for various applications and interactions.

## Contributing 🤝

Contributions and improvements are welcome! If you have ideas for enhancing gesture recognition or want to expand its capabilities, feel free to open issues or submit pull requests.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Let's enhance human-computer interaction and communication through gesture recognition technology! 🌟🖐️

For any questions or feedback, please don't hesitate to reach out.
