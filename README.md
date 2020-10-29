# Character Recognition for the Punjabi Language
*A TensorFlow implementation of a character recognition system for the Punjabi language (Gurmukhi script).*

These files comprise a character recognition system for the Punjabi language, more specifically for the Gurmukhi script. This is a challenging task due to the complex structural properties of the script. In fact, so far the model is trained only on the first five characters of the script. In the near future, it will be extended to the entire script. The task is also challenging due to the lack of easily available public datasets. As a result, we have constructed our own training dataset for this model.

## Description of the model

In our system, character is represented by a 10 x 10 matrix of 0's and 1's. The file preprocess-new-input-character.py contains code which can take as input a new image, 'char.png', and converts it into this representation. It does so by converting the image into grayscale, segmenting the image into a 10 x 10 grid and then replacing a rectangle in the grid with a 1 exactly when the difference between the least and most bright pixels within that rectangle is sufficiently large. As we will see below, the 100 grid pieces are the features of an input image on which the neural network is trained.

For example, the following image of a handwritten *sasa*

<img src="https://user-images.githubusercontent.com/26696492/97513252-fb562a80-1948-11eb-96d8-7b3795591fdc.jpg" height="64" width="85">

is converted into the following matrix:

```
[1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
[0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
[0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
[0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
[1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
[1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
[1, 1, 1, 1, 0, 0, 1, 0, 0, 0]
[0, 1, 1, 0, 0, 0, 1, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
```

As another example, the following handwritten *haha*

<img src="https://user-images.githubusercontent.com/26696492/97513275-05782900-1949-11eb-902b-b3d26bdf3ac5.jpg" height="75" width="87">

is converted into the following matrix:

```
[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
[0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
``` 

On the model, the neural network architecture is as follows. The input layer has 100 input units, there is one hidden layer with 100 hidden units and the output layer has 5 units. The input and hidden layers have tanh activation, while the output layer has softmax activation. We choose 100 input and hidden units so that the model may be trained on and learn the 100 grid pieces into which an input image is segmented.

## Requirements

* TensorFlow
* NumPy
* Matplotlib

## Data

The training dataset was assembled manually by the author and may be found in the file training-data.py. It consists of 100 characters, 20 each of *oora*, *ara*, *eeri*, *sasa* and *haha*, the first five characters of the Gurmukhi script.

## Training and using the model

The model is trained using stochastic gradient descent, with a learning rate of 0.01. It achieves >95% accuracy on the test set after ~10 epochs. To construct and train the model for yourself, run the file model.py. The file also includes tests for the input *sasa* and *haha* examples above. They are both classified correctly -- typical output probabilities for the test *sasa* are

```[0.16284972, 0.20005187, 0.20516102, 0.23086476, 0.20107257]```

and *sasa* is the fourth character, while typical output probabilities for the test *haha* are

```[0.13078634, 0.05030927, 0.19478032, 0.24436176, 0.37976232]```

and *haha* is the fifth character. Note also that the *haha* is more confidently classified by the model, while the *sasa* is less confidently classified. This is exactly as one would expect given the relative complexity of the two characters.
 
## Credits and Contact

This code was written by Gurtek Singh Gill. To ask questions or report issues, please send an email to rickygill01@gmail.com.
