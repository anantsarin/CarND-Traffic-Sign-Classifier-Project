# **Traffic Sign Recognition**

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"

[image4]: ./test_images_from_internet/7a.jpg "Traffic Sign 1"
[image5]: ./test_images_from_internet/1a.jpg "Traffic Sign 2"
[image6]: ./test_images_from_internet/3a.jpg "Traffic Sign 3"
[image7]: ./test_images_from_internet/2a.jpg "Traffic Sign 4"
[image8]: ./test_images_from_internet/6a.jpg "Traffic Sign 5"
[image9]: ./test_images_from_internet/5a.jpg "Traffic Sign 6"
[image10]: ./test_images_from_internet/4a.jpg "Traffic Sign 7"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

I used the data given in the "../data/*" of the workspace

* The size of training set is ?
    The size of the data set is divided into 3 parts
    * Training set 34799 images
    * Validation set:  4410 images
    * and then the test set: 12630 images
* The size of the validation set is ?
    Validation set:  4410 images
* The size of test set is ?
    test set: 12630 images
* The shape of a traffic sign image is ?
    Image data shape = (32, 32, 3)
* The number of unique classes/labels in the data set is ?
    number of unique classes/labels: 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed in our dataset
Since we have few classes with very little data as compared to the other classes mostly in the range of label 20  to  40. that's why we need more data for these classes.

* we could generate more data by changing the color of the images or by rotating the images,  But i didn't use any of the methods, As I just doubled the training data we are given for the traffic signs.

Now we have 69598 images in our training set
the second histogram can be seen in block [5] of the notebook



![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert colored images to  grayscale images this led us to the training data size of (69598, 32, 32, 1)
As per what I saw while processing the data multiple times, it also helped the model to learn faster as it surely decreases  data points
Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

I didn't normalize the data as I found that after normalizing my training set accuracy become less than what I had before


I decided to generate additional data because I want my training model to learn on more data to reach a better accuracy

To add more data to the data set, I just doubled the training data we are given for the traffic signs. this created a bigger data set hence more learning

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following
*  the data set is in grayscale
*  Size of the training data doubled for every class


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   					|
| Convolution 1 5x5     | 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 2 5x5     | 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten       	    | to 400 inputs.      							|
| Fully connected 1		| output 200   									|
| RELU					|												|
| dropout layer			|	with keep_prob = 0.5						|
| Fully connected 2		| output 120   									|
| RELU					|												|
| dropout layer			|	with keep_prob = 0.5						|
| Fully connected 3		| output 84   									|
| RELU					|												|
| Fully connected 4		| output 43   									|
| Softmax				| gives 42 output based on the last result		|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the approach shown the exercises before this project.
* I am using an Adam optimizer for the training the model
* learning rate of 0.001
* we divided our training set into small batches of 128 images per data set
* We are using 100 Epochs for training the data
* also we are using dropout layer keep_prob as 0.5 while learning on a training data set



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9 %
* validation set accuracy of 0.947 that is 94.7%
* test set accuracy of 92.9% ~ 93%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
        The first architecture I choose was Simple Let Architecture as it was tested by me before in the previous exercise which gave me a validation accuracy of 89%. As this was expected from the instructors.
* What were some problems with the initial architecture?
     Problems with the architectures were:
     * validation accuracy was very low
     * Test accuracy was even low
     * Also as per the histogram there was not equivalent data for all classes to learn
     * There were no Dropout layers in it which can make few of the layers to have weights as 0s

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function, or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. High accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Few of the points for why I adjusted the architecture to make it what it is now are:
	* Test accuracy was less when I used plain LeNet architecture with over fitting results
	* After adding 1 layer of dropouts with 0.5 the validation accuracy increased by 2 %
	* Second layer of dropout only increased the accuracy of 0.5 percent
	* Even after this the accuracy didn't reach the percent we needed that's why I added one more layer of FC
	* this layer of FC will take 200 inputs and 120 outputs
	* After adding this layer our validation accuracy moved t0 94.7%

* Which parameters were tuned? How were they adjusted and why?
	* I tuned few parameters to get better accuracy and found different results shown below:
		- learning rate
			+ 0.0009 accuracy 94%
			+ 0.002 accuracy 92%
			+ 0.001 accuracy 94.7%
		- Epochs
			+ 25 accu 91.2%
			+ 60 accu 93.1%
			+ 75 accu 92.8%
			+ 100 accu 94.7%
		- dropouts
			+ 0.8 accu 90 %
			+ 0.5 accu 94.7%

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
	- As Described before the LeNet models have proven good choice over many problems
	- tweaking it according to our need will provide a better result
	- I added a full FC layer because I think that the number of inputs as now in a considerable range that is 400 and if I add a new convolution layer it will decrease the inputs less that want we have
	- Reductio in inputs can cause loss of data hence FC layer is a better choice for me
	- For dropout layers help the model to learn from all the weights so that we will not have any weights as 0 which will make that branch of inputs useless



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10]


Most of the images are of different sizes and we need to pre-process the size to 32X32 that will fit in LeNet architecture.
Some of the images classes have less data hence these images are harder to classify when a new image with different from the dataset is given

The fifth image might be difficult to classify because the image is very blurry and when the pixels are divided as features they can take a different shape. Also in the learning data, we have very little data for class 25.

As we can see the accuracy we have are 85.7% accurate that means all other images are classified correctly except 1. As we saw later this will turn out to be only class 25 was not identifies correctly

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:



| Image										|     Prediction	        					|
|:-----------------------------------------:|:---------------------------------------------:|
|Speed limit (60km/h) 						|Speed limit (60km/h)							|
|Right-of-way at the next intersection 		|Right-of-way at the next intersection			|
|General caution 							|General caution								|
|No entry 									|No entry										|
|Road work 									|Speed limit (30km/h)							|
|Ahead only 								|Ahead only										|
|Wild animals crossing 						|Wild animals crossing							|

The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 85.7%. This compares favorably to the accuracy on the test set of 92.9% as we can see most of the images are classified correctly in the test set also which says only 7% of the images are not correctly classified.

Suggestions:

*  We need more data for the classes with less number of images to make there classification better
*  images should be preprocessed to fit in the model correctly

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the second last cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five softmax probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1         			| Speed limit (60km/h)   						|
| 1     				| Right-of-way at the next intersection			|
| 1						| General caution								|
| 1	      				| No entry										|
| 0.6				    | Speed limit (30km/h)							|
| 1				    	| Ahead only									|
| 0.9				    | Wild animals crossing							|

for the 5th image which our model didn't predict correctly probability is 0.6 and if we look at the top five probabilities [  6.01560056e-01,   1.82098046e-01,   7.32854158e-02, 3.67344730e-02,   1.75017789e-02] and there predictions [ 1,  2, 25, 28,  4] for our correct output still have a probability of .07 which means we are close to correctly predicting the image we just need better and more data for few classes

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

In the optional section, I tried to fetch all the convolution set, and as we can see the convolution and max-pooling helped me to see the how my network is trying to look into different parts of the image as it starts with the pixels and then comes the edges and filled areas and at the end, we are looking at the final complete pictures. this suggests that while doing convolutions we are looking at small features to build complete images in the final output.
The outputs can be seen in the last section of the IPython notebook for the 60 km/hr road sign.
