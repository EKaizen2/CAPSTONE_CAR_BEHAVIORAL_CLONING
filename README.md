NOTE: You have to downlaod the Udacity Self Driving nano degree engineer simulator before you can run this project
 [LINK GO DOWNLOAD SIMULATOR](https://github.com/udacity/self-driving-car-sim)

## Project Description

In this project, I use a neural network to clone car driving behavior.  It is a supervised regression problem between the car steering angles and the road images in front of a car.  

Those images were taken from three different camera angles (from the center, the left and the right of the car).  

The network is based on [The NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), which has been proven to work in this problem domain.

As image processing is involved, the model is using convolutional layers for automated feature engineering.  

### Files included

- `model.py` The script used to create and train the model.
- `drive.py` The script to drive the car. You can feel free to resubmit the original `drive.py` or make modifications and submit your modified version.
- `utils.py` The script to provide useful functionalities (i.e. image preprocessing and augumentation)
- `model.h5` The model weights.
- `environments.yml` conda environment (Use TensorFlow without GPU)
- `environments-gpu.yml` conda environment (Use TensorFlow with GPU)

Note: drive.py is originally from [the Udacity Behavioral Cloning project GitHub](https://github.com/udacity/CarND-Behavioral-Cloning-P3) but it has been modified to control the throttle.


### Install required python libraries before you can Run this project:

You need an [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html) to use the environment setting.

When you have Anaconda or Mini-conda install on your system, run one to the codes below to create an environment for the Simulator
```python
# Use TensorFlow without GPU
conda env create -f environment.yml 

# Use TensorFlow with GPU
conda env create -f environment-gpu.yml
```

Or you can manually install the required libraries (see the contents of the environment*.yml files) using pip.

### Run the pretrained model

Start up the SIMULATOR, choose a scene and press the Autonomous Mode button.  Then, run the model as follows in command line:

```python
python drive.py model.h5
```

### To train the model yourself

You'll need the data folder which contains the training images(You will need to drive the model in manual mode on the simulator to gather data for training the model).

```python
python model.py
```

This will generate a file `model-<epoch>.h5` whenever the performance in the epoch is better than the previous best.  For example, the first epoch will generate a file called `model-000.h5`.


## References
- NVIDIA model: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
- Udacity Self-Driving Car Simulator: https://github.com/udacity/self-driving-car-sim
