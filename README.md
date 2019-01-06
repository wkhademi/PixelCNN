# PixelCNN
A TensorFlow implementation of the PixelCNN.

## Running the Model
There are three different datasets that the model is intended to be tested on: MNIST, Frey Face, and CIFAR-10 dataset.  
To train and test the model, run the command: `python main.py [--MNIST | --FREY | --CIFAR]`.

## Results
### Binarized MNIST
The model was trained for 25 epochs on a binarized version of the MNIST dataset.  
The model was able to reach a negative log-likelihood score of 80.97 nats.

**Incomplete Images:**
![alt text](https://github.com/wkhademi/PixelCNN/blob/master/images/MNIST_Incomplete.png "MNIST Incomplete")

**Completed Images:**
![alt text](https://github.com/wkhademi/PixelCNN/blob/master/images/MNIST_Completed.png "MNIST Completed")

### Frey Face
**Incomplete Images:**
![alt text](https://github.com/wkhademi/PixelCNN/blob/master/images/FREY_Incomplete.png "FREY Incomplete")

**Completed Images (Current):**
![alt text](https://github.com/wkhademi/PixelCNN/blob/master/images/FREY_Completed.png "FREY Completed")

### CIFAR-10
*Currently a work in progress...*

## Original Paper
The original PixelCNN paper was written by Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu.  
The paper can be found here: [Pixel Recurrent Neural Networks](https://arxiv.org/pdf/1601.06759.pdf).
