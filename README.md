# LITS_Hybrid_Comp_Net  (Paper Link - https://arxiv.org/pdf/1909.04797.pdf)
## Built With/Things Needed to implement experiments

* [Python](https://www.python.org/downloads/) - Python-2 
* [Keras](http://www.keras.io) - Deep Learning Framework used
* [Numpy](http://www.numpy.org/) - Numpy
* [Sklearn](http://scikit-learn.org/stable/install.html) - Scipy/Sklearn/Scikit-learn
* [CUDA](https://developer.nvidia.com/cuda-80-ga2-download-archive) - CUDA-8
* [CUDNN](https://developer.nvidia.com/rdp/assets/cudnn_library-pdf-5prod) - CUDNN-5 You have to register to get access to CUDNN
* [LiTS](https://competitions.codalab.org/competitions/17094) - LiTS-dataset website
* [12 gb TitanX]- Used for all networks mentioned here
* [OpenCV] - To visualize 2D slices. Can be installed using Pip
* Requires X forwarding and MatplotLib to open the interactive Histogram Distribution clicker 
* [Nibabel](https://nipy.org/nibabel/) - To open nii images. Can be installed using Pip

## Preprocessing 
### Run Preprocess_create_numpy_array_from_data.py to obtain the 3d numpy arrays of the subjects Controls are mentioned in the file itself in the initial comments but to summarize please read the following
 * Set filepath to load data
 * Set filepath to save the 2D slices as .png to visualize them
 * Set filepath to save the numpy arrays 
 * The for loop of line 58 can help you control the files you need changes on
 * The values of the numpy array are normalized between 0-255 from initial values and not 0-1 and has to be made 0-1 before being passed to the network. The value is kept between 0-255 to easily visualize as png but for the network please use 0-1 

### When the code is run it will pop up an Intensity Distribution and the aim is to select the rightmost peak of the graph as shown below -

![alt text](https://github.com/raun1/LITS_Hybrid_Comp_Net/tree/master/fig/img_1.png)


### The following images show the preprocessed files resulting from this method - 

![alt text](https://github.com/raun1/LITS_Hybrid_Comp_Net/tree/master/fig/img_2.png)


### You will notice that the rightmost peak intensity varies across the subject and this is expected.

### If you have difficulty to obtain the right most peak, you may redo that specific file using the for loop control in line 58.

### In the event this is too troublesome a simple fix is to normalize the subject by setting all pixels below -200 to 0 and all pixels over 250 to 0 and segmenting the liver region seperately first. Following this the Distribution graph returns only one peak and it is easy to select it.

### The function def find_min_and_max(img) in line 25 is the fuction which actually does the left and right range selections and can be reused as desired.



## 2D Network to segment liver region and also to predict large tumors 

### This network is based on the idea of the Complementary Networks presented in the paper here - https://arxiv.org/abs/1804.00521. The exact architecture details are mentioned in the paper - https://arxiv.org/pdf/1909.04797.pdf in section 2.3. The Tumor detection Networks has to be compiled and trained twice.
* The 2D CompNet for the liver segmentation is trained for 40 epochs using the Adam optimizer with a learning rate of 5e-5
* Train the networks using the Adam optimizer with a learning rate of 5e-5, and having an early stopping scheme with the tolerance being set to 5; then we train the networks with a learning rate of 1e-6 using an early stopping with a tolerance of 10 trials. Both steps have 150 maximum number of epochs for training.

## 3D Network to segment small tumors

### This network is a 3d version working on small 3D patches and exact details of the architecture is mentioned in section 2.3 in https://arxiv.org/pdf/1909.04797.pdf. The Tumor detection Networks has to be compiled and trained twice.

* Train the network using the Adam optimizer with a learning rate of 5e-5, and having an early stopping scheme with the tolerance being set to 5; then we train the networks with a learning rate of 1e-6 using an early stopping with a tolerance of 10 trials. Both steps have 150 maximum number of epochs for training.


email me - rd31879@uga.edu for any questions !! Am happy to discuss 



