# Cloud Texture Classifier
## Application implementing a cloud texture classifier based on different neural network architectures

## Introduction

Cloud classification represents an incredibly useful meteorological practice, with significant applications in various fields such as aviation, agriculture, and more commonly, weather forecasting. However, cloud classification is a process requiring the opinion of a meteorology expert and therefore, it is prone to a degree of human error and dependent.

The scope of this application is the automatization of the cloud classification process based on their texture via artificial intelligence solutions.

## Application Description

The application develops a classifier where the classification criteria is determined by cloud texture, based on the artificial neural network technologies. This problem is tackled in two different ways, by designing two different classifiers:
- A standard classifier using Convolutional Neural Networks (CNN) architecture, which employs special neuron layers called convolutional filters to perform convolutions across the database images and extract the relevant feature pattern;
- An experimental, more recent classifier using simple Artificial Neural Netowrk (ANN) architecture in which the input database of images is refined into individual vectors of texture codes using Local Ternary Patterns.

Explaining the repository contents:
- kaggle folder contains **training_classifiers** and **testing_classifiers** notebooks developed using the Kaggle IDE, used for training the designed architectures and test already identified optimal models, respectively;
- google colab folder contains the **file_extraction_procedure** notebook developed using the Google Colab IDE and represents the code used for extracting textures out of images (more on this aspect in a second), the code is purely demonstrative and does not function on its own ( however, following instructions written in the notebook, it can be configured to run successfully on your personal Google Drive)

Most of the resources have already been made publicly and will be listed here:
- training_classifiers and testing_classifiers can already be found publicly on kaggle <a href ="https://www.kaggle.com/code/mateiciulei/training-classifiers" target="_blank"> here (training) </a> and <a href = "https://www.kaggle.com/code/mateiciulei/testing-classifiers/edit" target = "_blank"> here (testing) </a>;
- the CCSN database of images used for this project can also be found <a href = "https://www.kaggle.com/datasets/mateiciulei/ccsn-public" target = "_blank"> here (CSSN_database) </a> (the version loaded on Kaggle);
- the alternative database of extracted textures can be found <a href = "https://www.kaggle.com/datasets/mateiciulei/descriptori-ltp-ccsn" target ="_blank"> here (LTP_database) </a>;
- the collection of different optimal models identified can be found <a href = "https://www.kaggle.com/models/mateiciulei/clasificatoare_nori/Keras/optiuni/1" target="_blank"> here (optimal_models) </a>;

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

Two architectures have been developed with the basic ANN classifiers sharing the same architecture, the only notable difference being the input shape according to the input vector used. Dropout layers were experimented with, but ultimately did not improve final results and as such both architectures retain only basic layers.

The CCSN database mentioned above identifies 11 object classes constisting of different cloud topologies: altocumulus (Ac), altostratus (As), cirrus (Ci), cirrostratus (Cs), cirrocumulus (Cc), cumulus (Cu), Cumulonimbus (Cb), Nimbostratus (Ns) , stratocumulus (Sc), stratus (St) and contrail (Ct) (short for condensation trail), the latter being an unconventional class type because it describes artificially formed clouds by aircraft engines.

The code application is split into two separate Jupyter notebooks: one for training an ANN model from scratch and one for evaluating the optimized versions of the ANN models. The performance evaluation is done by first plotting relevant accuracy/loss graphs during the training phase and then calculating performance metrics of a given classifier on test time, meant to measure its efficiency, as well as plot the associated confusion matrix. The system calculates per-class, as well as micro- and macro-averaged values of Precision, Recall and F1-score. The accuracy/loss graphs are not stored and as such are exclusive to the session of the training notebook.

The best version of the algorithm developed for extracting the cloud textures is computationally expensive ( $$O(n^2)$$ complexity due to the fact that the whole image needs to be iterated over and the vast majority of the database images are 400x400 pixels in size) and each run took roughly five hours per code texture extraction. The code textures needed to be pre-emptively stored for ease of access and loading into memory before traning and testing runtimes. 

This was done so by developing a separate file extraction procedure in Google Colaboratory, where I mounted my personal Google Drive containing the already loaded up raw database of images and extracting each class to a specific noise threshold file, formatted as binary. By doing so, squbsequently, the time required for loading the appropiate code texture database into memory shortened itself down to half a minute.

The development platform chosen for this project was Kaggle, due to its Machine Learning-oriented features, which helped particularly during the training phases of the CNN classifier via graphical accelerators. This was used in conjuction with Google Colaboratory, which served exclusively towards the extraction of the texture descriptors because of a much more accessible file system and lenient timeout sessions.

Libraries used for this application:
- **tensorflow**, namely **keras**, used for constructing both model infrastructure and affiliate data pipelines, as well as importing/exporting models;
- **matplotlibpyplot**, used for charting the classifiers' performances;
- **numpy**, used for data shuffling and debugging;
- **sklearn**, used for calculating performance metrics and data scaling;
- **os** and **cv2** libraries for browsing and accessing input data.

## Application Showcase

The following sections report the performances obtained by:
- the CNN-based classifier analyzing a database of raw cloud RGB images; 
- standard ANN-based classifier analyzing a database of base texture descriptors extracted for noise threshold of value 5;

Architectures of discussed classifiers can be consulted in the training notebook. All architectures employ a Softmax classifier on the last layer for predicting trust intervals. The CNN classifier is trained over 40 epochs, while the LTP base texture classifier is trained over 80. Starting learning rate is 0.0001 and is adjusted via the Adam optimizer.

### CNN classifier

The designated CNN architecture showed exceptional results during training, boasting a less than 5% rate of error and a relatively modest model size for its caliber. 
<div align = "center">
  
| Validation Accuracy [%] | Train time [s] | Validation Loss | Model Size [params/MB] |
|:-----------------------:|:--------------:|:---------------:|:-----------------------:|
|   95.31                 |   159          |  0.244          |  $$4.73*10^6$$ / 18.04  |
  
   <br>
   <p align ="center"><i> Optimal CNN classifier training performance table </i></p>
</div>
<br>

By charting the training performance graphs, we can notice that the classifier achieves proper generalization of the cloud texture features, while also maintaining a good rate of convergence towards the optimal solution.

<br>
<div align="center">
  <img src="https://github.com/user-attachments/assets/0badb2d2-86d7-4359-85c2-7cce49247c46" alt="train_CNN" width = "712" height = "290"/>
  <p align ="center"><i> Training performances of the CNN classifier </i></p>
</div>

Taking it a step further, the confusion matrix communicates valuable information as to where a mistake would occur during erroneous prediction cases:

<br>
<div align="center">
  <img src="https://github.com/user-attachments/assets/1766512d-f0e5-4a5b-b1dd-505cec1624f3" alt="conf_matrix_CNN" width = "800" height = "382"/>
  <p align ="center"><i> Confusion matrix for the optimal CNN classifier </i></p>
</div>
<br>

There seems to be no clear pattern of confusion, and the reduced scale of the database (2543 images) as well as the test set, deem this analysis inconclusive. The varying percentages of mispredicted examples in each class category, even though the effective number of mispredicted samples is the same, are determined by the size of a particular class. Intuitively, classes with fewer test samples will give more weight per test sample when evaluating performance metrics. This inter-class imbalance is addressed when computing micro- and macro- averaged performance metrics.

<br>
<div align="center">
  <img src="https://github.com/user-attachments/assets/8e3c09c5-9fc0-4b5e-8034-3f8063d8f62d" alt="class_CNN"/>
  <p align ="center"><i> Performance table evaluating CNN metrics per class<br>(as shown in console output) </i></p>
</div>
<br>

F1-score is used as a measure of definitive efficacy of the analysed classifier, whether we're evaluating it class-wise or in general. As it can be observed, most classes report prediction efficencies higher than 97% in the case of the CNN classifier.

<br>
<div align="center">
  <img src="https://github.com/user-attachments/assets/285a4511-e6b2-48ed-ae65-aa8fbe836df0" alt="micro_macro_CNN"/>
  <p align ="center"><i> Performance table evaluating CNN micro- and macro-averaged metrics<br>(as shown in console output) </i></p>
</div>
<br>

Micro- and macro- averaged values are used in order to quantify the general efficacy of the classifier, and as mentioned before, they handle class imbalances. Micro-averaged values better reflect the efficacy of the classes with more samples, while macro-averaged values better reflect the efficacy of the classes with fewer samples. Due to the fact that the test samples are relatively evenly distributed across all classes, the differences in values show on a much smaller scale, hence the invariance present in the values above. However, demonstrating this is beyond the scope of quantifying performance.

### Base LTP texture classifier

Moving on to the base LTP texture classifier, with extracted texture codes given a noise threshold value of 5, the system performs poorly, at best. Preprocessing techniques such as database normalization and standardization did not report an increase in performance. During training, there have been attempts to improve performance architecture-wise by fine-tuning parameters as well as integrating BatchNormalization and Dropout layers, with little effect, going as far as being counterproductive at times. The optimal model version has been documented below.

<div align = "center">
  
| Validation Accuracy [%] | Train time [s] | Validation Loss | Model Size [params/KB]   |
|:-----------------------:|:--------------:|:---------------:|:------------------------:|
|   26.71                 |   2.8          |  2.069          |  101,435 / 396.23        |
  
   <br>
   <p align ="center"><i> Optimal LTP texture classifier (noise threshold 5) training performance table </i></p>
</div>
<br>

By further analysing the training performances, we can notice that the system showcases a rather poor convergence rate and an inability to achieve proper generalization of the classified texture codes.

<br>
<div align="center">
  <img src="https://github.com/user-attachments/assets/995119a6-d903-451c-88c7-9f09cd023b93" alt="train_LTP" width = "712" height = "290"/>
  <p align ="center"><i> Training performances of the optimal LTP texture classifier (noise threshold 5) </i></p>
</div>

These reports are relevant for all code texture classifiers, both base and rotation invariant, with slight performance differences. Although per-class performances may vary strongly depending on the noise threshold value selected, all classifiers report roughly the same micro- and macro- averaged values.

<br>
<div align="center">
  <img src="https://github.com/user-attachments/assets/abde5637-029a-49bd-af1a-e5805a43e86a" alt="conf_matrix_LTP" width = "800" height = "382"/>
  <p align ="center"><i> Confusion matrix for the optimal LTP texture classifier (noise threshold 5) </i></p>
</div>
<br>

As it can be seen above, the noise threshold of value 5 LTP base classifier shows a strong bias towards class Sc and the vast majority of mispredictions is directly tied to class Sc.

<br>
<div align="center">
  <img src="https://github.com/user-attachments/assets/fb82c4ae-db32-4e10-8a03-752e1300fdee" alt="class_CNN" width = "410" height="481"/>
  <p align ="center"><i> Performance table evaluating LTP metrics per class<br>(as shown in console output) </i></p>
</div>
<br>

The inability to develop proper generalization of features is even more so outlined in the per-class performance table shown above, with half of the classes reporting no classifying efficacy whatsoever.

<br>
<div align="center">
  <img src="https://github.com/user-attachments/assets/5eeddf7f-2d23-4dd7-9de1-a3efc4cc8f63" alt="micro_macro_CNN" />
  <p align ="center"><i> Performance table evaluating LTP micro- and macro-averaged metrics<br>(as shown in console output) </i></p>
</div>
<br>

## Conclusions

Before proceeding, I want to mention that all proper testing has been conducted on the developed code to ensure the proper functioning of the presented extraction algorithms, so that these conclusions may be valid.

The needs of this coding application have already been met by the CNN classifier, which declared exceptional results. However, the LTP base texture classifier might represent a much better long term solution, should the texture extraction algorithm be optimized. Below I have presented two tables showcasing the pros and cons of both models discussed.

<div align = "center">
  <p align ="center"><b> CNN Classifier Evaluation </b></p>
  
| Pros | Cons | 
|:----|:----|
| exceptional performance            | long runtimes
| robust, well-documented technology | extensive computational resources required
| beginner-friendly                  | very large in size

</div>
<br>

<div align = "center">
  <p align ="center"><b> LTP Classifier Evaluation <br> (noise threshold 5) </b></p>
  
| Pros | Cons | 
|:----|:----|
| exceptional runtimes on basic hardware           | performs poorly
| compact size                                     | requires expert fine-tuning and <br> preprocessing techniques
| analysed data is also reduced in size            | texture extraction requires optimization

</div>
<br>

The model size of the LTP classifier is even further reduced if we consider its rotation invariant counterpart, down to just 66KB, which makes it even more desirable for future applications. However, there is the fatal issue of performance.

My best educated guess is that a lot of the database's coherence is compromised when the code is being split into positive and negative code textures, as any ternary LTP code texture containing exclusively positive or negative values are all converted to void values. However, this process was implemented according to scientific documentation consulted, so I am assuming that the database needs to be subjected to preprocessing techniques far better than what I can provide. 

One solution would be to never split the code at all and implement a histogram based on the ternary 8-"bit" code, but that would mean each texture descriptor would have to store a total of $$3^8=6561$$ possible 8-"bit" texture codes, which complicates memory allocation a lot and it would more efficient to just switch to a CNN configuration on the raw images at that point.

In summary, the CNN classifier has more than met the standards required for this project, but the LTP classifier presents a lot of potential for research and future applications.

## References
1. Ojala, T., Pietikäinen, M., senior member, IEEE, Mäenpää, T., *Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns*, *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 24, no. 7/2002, pp. 971-987 <a href = "https://ieeexplore.ieee.org/abstract/document/1017623" target="_blank"> (link)</a>
2. Tan, X., Triggs, B., *Enhanced Local Texture Feature Sets for Face Recognition Under Difficult Lighting Conditions*, *IEEE Transactions on Image Processing*, vol. 9, no. 6/2010, pp. 1635-1650 <a href = "https://ieeexplore.ieee.org/abstract/document/5411802" target = "_blank"> (link)</a>
3.  Zhang, J. L., Liu, P., Zhang, F., & Song, Q. Q. (2018), *CloudNet: Ground-based cloud classification with deep convolutional neural network.*, *Geophysical Research Letters*, vol. 45, pp. 8665–8672.
 <a href="https://doi.org/10.1029/2018GL077787" target="_blank"> (link)</a>
