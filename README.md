# Trained CNN for Genre classification
Trained CNN model for Genre classification on GTZAN dataset [CNN Model: https://github.com/Hguimaraes/gtzan.keras]

## Test trained CNN model
 In the `./weights/` you can find trained model weights and model architecture.

## Dependencies
 * [Keras](https://keras.io) or [PyTorch](http://pytorch.org)
 * numpy
 * librosa - for audio feature extraction
 
 ## Accuracy

 * Training (at Epoch 35):
    
    * Training loss:    0.1283
    
    * Training accuracy:    0.9596

 * Testing:
    
    * Test loss:    0.3936
    
    * Test accuracy:    0.8921

## Accuracy, Loss and Confusion matrix
![alt text](./images/accuracy_and_loss_vgg16.png "VGG16 Model")
![alt text](./images/confusion_matrix_vgg16.png "Confusion Matrix of the VGG16 Model")

## License
MIT License
