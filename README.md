# Trained CNN for Genre classification

![](https://img.shields.io/badge/-status:wip-5319e7.svg)
![](https://img.shields.io/github/license/NazarPonochevnyi/Trained-CNN-for-Genre-classification)
![](https://img.shields.io/github/languages/code-size/NazarPonochevnyi/Trained-CNN-for-Genre-classification)
![](https://img.shields.io/github/last-commit/NazarPonochevnyi/Trained-CNN-for-Genre-classification)

🎵 Trained CNN model for Genre classification on GTZAN dataset [CNN Model: https://github.com/Hguimaraes/gtzan.keras]

## Test trained CNN model
 In the `./weights/` you can find trained model weights and model architecture.
 Also, you can download .h5 file and place it manually to ./weights/ directory using this <a href="https://drive.google.com/file/d/1rJEw1N--pgX4w40yQfsuqYiD8lTR3Hz4/view?usp=sharing" target="_blank">link</a>.
 
 For test this trained model, you can run `python3 predict_vgg16.py`.
 
 Also, if you want to test your custom song or turn off debug messages, you can change code in the `get_genre()`'s function arguments.
 For example, you can input your song path instead default `./audios/classical_music.mp3` path or toggle next boolean function's argument to turn off debug messages.

## Overview
For train CNN model of deep learning:

1. Read the audios as melspectrograms, spliting then into 3s windows with 50% overlaping resulting in a dataset with the size 19000x129x128x1 (samples x time x frequency x channels)**.
2. Shuffle the input and split into train and test (70%/30%)
3. Train the CNN and validate using the validation dataset

** In the case of the VGG, the channel need to have 3 channels

## Dependencies
 * [Keras](https://keras.io)
 * [Numpy](http://www.numpy.org)
 * [Librosa](https://librosa.github.io) - for audio feature extraction
 
 ## Accuracy

 * Training (at Epoch 35):
    
    * Training loss:    0.1283
    
    * Training accuracy:    0.9596

 * Testing:
    
    * Test loss:    0.3936
    
    * Test accuracy:    0.8921

## Accuracy, Loss and Confusion matrix graphs
![alt text](./images/accuracy_and_loss_vgg16.png "VGG16 Model")
![alt text](./images/confusion_matrix_vgg16.png "Confusion Matrix of the VGG16 Model")

## License
[MIT Licence](./LICENSE)
