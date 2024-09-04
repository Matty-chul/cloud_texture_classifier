# Cloud Texture Classifier
## Application implementing a cloud texture classifier based on different neural network architectures

## Introduction

Cloud classification represents an incredibly useful meteorological practice, with significant applications in various fields such as aviation, agriculture, and more commonly, weather forecasting. However, cloud classification is a process requiring the opinion of a meteorology expert and therefore, it is prone to a degree of human error and dependent.

The scope of this application is the automatization of the cloud classification process based on their texture via artificial intelligence solutions.

## Application Description

The application develops a classifier where the classification criteria is determined by cloud texture, based on the artificial neural network technologies. This problem is tackled in two different ways, by designing two different classifiers:
- A standard classifier using Convolutional Neural Networks (CNN) architecture, which employs special neuron layers called convolutional filters to perform convolutions across the database images and extract the relevant feature pattern;
- An experimental, more recent classifier using simple Artificial Neural Netowrk (ANN) architecture in which the input database of images is refined into individual vectors of texture codes using Local Ternary Patterns.

### Extracting Textures Using Local Ternary Patterns

Local Ternary Patterns represent the upgraded, noise-resistant version of Local Binary Patterns. Extracting textures using Local Ternary Patterns involves iterating over all the pixels (excluding borders) within an image and calculating individual texture codes over a given fixed spatial region surrounding the reference pixel, most commonly the 8 contiguous pixels. The texture codes are calculated by substracting the value of the reference pixels from the neighboring pixels and by filtering these results through a three-branch comparator function factoring in a flat noise resistance value, each neighboring pixel gets assigned a value of either 1, 0 or -1. 

By assigning a standard direction for interpretation, each reference pixel can extract an 8-bit code using a ternary alphabet {-1,0,1}. These codes are then broken down into two standard binary codes in the following manner:
1. A **"positive"** binary code storing the original code orientation with the negative values converted to 0;
2. A **"negative"** binary code storing the original code orientation with the positive values converted to 0, while negative values get converted to their positive counterparts.

<br>
<div align="center">
  <img src="https://github.com/user-attachments/assets/24f8756e-f027-4520-a36a-a10183aeee95" alt="code-conv"/>
  <p align ="center"><i> Figure describing the code splitting operation </i></p>
</div>
<br>


These codes are ultimately stored as integer values ranging from 0 to 255 in a histogram, representing the texture descriptor for the analyzed image. The cumulative collection of texture descriptors encapsulates the input feed for the experimental ANN classifier.

### Extracting Rotation Invariant Textures

Furthermore, rotation invariant texture descriptors can be extracted by pefrorming bitwise operations on the extracted binary codes. To understand the relevance of this task, take the splotch on a ladybug wing, for example: it could be oriented to the left, oriented to the right, flipped upside down, each of these identified textures presenting different code values, but all these variations describe the same texture shape, a small black dot. Through applying rotation invariance texture extraction to an image, we can identify all texture code recurring patterns, regardless of the angle at which they were rotated.

To achieve rotation invariance for a particular texture, one needs to replace the decimal value of the current texture with the bit configuration presenting the minimal value. 

Let's consider the positive code in the image above. Starting from the top right corner, the identified texture code is **00010010**, or **18** in decimal. By performing circular shift bits to the right we get the following values **{ 9, 18, 33, 66, 132}** out of which **9** is the lowest and is therefore the rotation invariant texture code identified for decimal number **18**.

Scientific literature identifies 36 such rotation invariant 8-bit code textures. Any decimal value ranging 0-255 identifies itself with a rotation invariant counterpart.

### Putting it all together in code

To sum up the theoretical notions presented above, the application compiles three different data pipelines for classification:
1. A data pipeline consisting of raw, unprocessed images from the Cirrus Cumulus Stratus Nimbus (CCSN) Database, which can be viewed <a href="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CADDPD" target="_blank">here</a>;
2. A secondary data pipeline consisting of the base texture descriptors associated with the images listed in the previous entry;
3. A third data pipeline consiting of rotation invariant texture descriptors;

<br>
<div align="center">
  <img src="https://github.com/user-attachments/assets/37fadce9-abf3-4bc0-943a-ab7a28a5c6f7" alt="descriptors" width = "640" height = "360"/>
  <p align ="center"><i> Figure describing the descriptors vector structure </i></p>
</div>
<br>

The image presented above illustrates the fashion in which the extracted code textures are stored in a vector. Each container represents a histogram of either positive or negative texture codes belonging to a specific color channel, denoted by the appropiate mathematical signs. Each subsequent histogram is offset by 256 for the base texture descriptor and 36 for the rotation invariant texture descriptor in order to allow contiguous storaging.

The code textures were extracted using Local Ternary Patterns for the following noise thresholds {0,2,5}, with the latter being the recommended value in documentation. This results in 3 different base texture databases. By factoring in their texture invariant counterparts and the original image database, the application compiles 7 different classifiers: one CNN-based classifier being trained on raw images and six other standard ANN-based classifiers being trained on refined texture descriptors.

Two architectures have been developed with the basic ANN classifiers sharing the same architecture, the only notable difference being the input shape according to the input vector used.

The CCSN database mentioned above identifies 11 object classes constisting of different cloud topologies: altocumulus (Ac), altostratus (As), cirrus (Ci), cirrostratus (Cs), cirrocumulus (Cc), cumulus (Cu), Cumulonimbus (Cb), Nimbostratus (Ns) , stratocumulus (Sc), stratus (St) and contrail (Ct) (short for condensation trail), the latter being an unconventional class type because it describes artificially formed clouds by aircraft engines.

The code application is split into two separate Jupyter notebooks: one for training an ANN model from scratch and one for evaluating the optimized versions of the ANN models. The performance evaluation is done by first plotting relevant accuracy/loss graphs during the training phase and then calculating performance metrics of a given classifier on test time, meant to measure its efficiency, as well as plot the associated confusion matrix. The system calculates per-class, as well as micro- and macro-averaged values of Precision, Recall and F1-score. The accuracy/loss graphs are not stored and as such are exclusive to the session of the training notebook.

The development platform chosen for this project was Kaggle, due to its Machine Learning-oriented features, which helped particularly during the training phases of the CNN classifier via graphical accelerators. This was used in conjuction with Google Colaboratory, which served exclusively towards the extraction of the texture descriptors because of a much more accessible file system and lenient timeout sessions.

Libraries used for this application:
- **tensorflow**, namely **keras**, used for constructing both model infrastructure and affiliate data pipelines, as well as importing/exporting models;
- **matplotlibpyplot**, used for charting the classifiers' performances;
- **numpy**, used for data shuffling and debugging;
- **sklearn**, used for calculating performance metrics and data scaling;
- **os** and **cv2** libraries for browsing and accessing input data.
