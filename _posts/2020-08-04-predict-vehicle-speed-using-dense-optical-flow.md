---
layout: post
title: Predict Vehicle Speed Using Dense Optical Flow
date: 2020-08-04
tags: computer-vision autonomous-vehicles
---

> Predict vehicle speed using dashcam videos and neural networks. Inspired by comma.ai speed challenge.


<!--more-->


I stumbled across the [comma.ai speed challenge](https://github.com/commaai/speedchallenge){: rel="nofollow"} so I decided to give it a shot. There are two dashcam videos: one video is used for training (20,400 frames @ 20 FPS) and the other video is used for testing (10,798 frames @ 20 FPS). The training video is accompanied by the ground truth speed at each frame. The objective is to predict the speed of the test video at each frame. I also managed to acquire a [third video](https://github.com/antoninodimaggio/Voof/blob/master/data/test/test.mp4) with the accompanying ground truth speeds which will serve as a test measure to see how well the methods demonstrated can generalize.

The [code](https://github.com/antoninodimaggio/Voof) can be found on my [GitHub](https://github.com/antoninodimaggio).

<iframe width="640" height="480" src="https://www.youtube.com/embed/vko7tUqESHU" frameborder="0"></iframe>{: style="padding-top: 25px;" class="center"}
<div align="center">
  Training video for the comma.ai speed challenge
</div>

## Overview
I approached the problem more so as a series of experiments as opposed to an actual attempt to get the lowest MSE on the test set. My approach is pretty simple. First calculate the optical flow field of successive images, then train a CNN on these optical flow fields to predict speed.

## Optical Flow
[Optical flow](https://en.wikipedia.org/wiki/Optical_flow){: rel="nofollow"} quantifies the apparent motion of objects between frames. Optical flow can also be defined as the distribution of apparent velocities of movement of brightness patterns in an image. Sparse optical flow constructs flow vectors for "interesting features" while dense optical flow constructs flow vectors for the whole frame. Sparse optical flow is more computationally efficient but less accurate. I am using dense optical flow which I will refer to as just optical flow from now on. Optical flow has historically been an optimization problem, however neural networks have been shown to work [better](https://arxiv.org/pdf/1612.01925.pdf){: rel="nofollow"} under certain conditions. Since there are two viable methods to calculate optical flow I decided to try out both. Calculating the relative velocity of the vehicle directly from an optical flow field requires [depth](https://www.youtube.com/watch?v=OB8RncJWIqc){: rel="nofollow"} (this was not obviously apparent to me at first). In my case the only way to estimate depth would be to use another neural network, which is not a method I chose to explore (although I believe that it may hold promise in terms of generalization).

## Video Preprocessing
With optical flow estimation in mind, the saturation of each pair of frames was augmented with the same uniform random variable to account for illumination changes that will severely deteriorate performance. This will also help with any overfitting issues that may arise. The majority of the sky and car hood are then cropped since they do not really change between successive frames. The frames are then resized and interpolated to work with the CNN architecture.
![Ilummination change]({{ '/assets/images/augmented_brightness.png' | relative_url }})

![Final transformed image]({{ '/assets/images/final.png' | relative_url }})

## Method #1: Gunnar-Farneback Dense Optical Flow
Gunnar-Farnebeck is an optimization based method for estimating dense optical flow. Two successive frames are preprocessed and fed into the algorithm. The resulting two-dimensional optical flow field can then be turned into a 3 channel RGB image via the following [method](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html){: rel="nofollow"}.
This process is repeated for each pair of successive frames. The resulting RGB optical flow images are accompanied by their appropriate ground truth speeds and saved as a PyTorch dataset. All of the code for this can be found in [preprocess_farneback.py](https://github.com/antoninodimaggio/Voof/blob/master/preprocess_farneback.py).

## Method #2: PWC-Net Dense Optical Flow
Training data to facilitate supervised learning for optical flow estimation neural networks is hard to come by. Animation turns out to be the solution to this problem since the transformations between each frame can be used as ground truths. I like using PyTorch so I modified this [PWC-Net implementation](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch){: rel="nofollow"} to work with modern PyTorch. I used the model, provided by the orginal authors, that was pre trained on the [MPI Sintel](http://sintel.is.tue.mpg.de/){: rel="nofollow"} dataset. PWC-Net is fairly complicated and requires a C++ extension to work with PyTorch, the best way to learn more would be to read the [paper](https://arxiv.org/abs/1709.02371){: rel="nofollow"}. The procedure is similar to that of Method #1, however the optical flow field is left in two-dimensions, since the output of PWC-Net is slightly different compared to that of Gunnar-Farneback. All of the code for this can be found in [preprocess_pwc.py](https://github.com/antoninodimaggio/Voof/blob/master/preprocess_pwc.py).

![PWC-Net architecture]({{ '/assets/images/pwc_arch.jpg' | relative_url }}){: style="padding-top: 50px; width: 100%;" class="center"}.

<div align="center">
  PWC-Net architecture
</div>

## CNN Training

### CNN Architecture
Once we have optical flow, we can then attempt to train a convolutional neural network to predict speed. Note that this is a regression task not a classification task (although it would be interesting to explore this problem as such). The CNN architecture is from the [End-to-End Deep Learning for Self-Driving Cars](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/){: rel="nofollow"} blog post by NVIDIA.

![NVIDIA CNN]({{ '/assets/images/nvidia_cnn.jpg' | relative_url }}){: style="padding-top: 50px; width: 50%;" class="center"}
<div align="center">
  NVIDIA CNN architecture
</div>
<br>
The model was implemented using PyTorch with the following hyperparameters:

* ` criterion = torch.nn.MSELoss() `
* ` optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) `
* ` scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
        patience=0, threshold=0.2, threshold_mode='abs', min_lr=1e-8) `

I also used Kaiming normal initialization for all of the convolutional layers which I found made the model converge faster.

### Training
The models for both methods were trained in a very similar fashion. With 80% of the data reserved for training and 20% of the data for evaluation. It is important to note that the data was not randomly shuffled, since this does not preserve integrity between the training and evaluation sets. I noticed that a bunch of previous work randomly shuffled the training and evaluation data, this method will inevitably leak information to the model due to the temporal nature of the dataset (ie. the model can be trained on a frame that came right before a frame in the evaluation dataset), tainting any results that they may have achieved. The Gunnar-Farneback model was trained for 20 epochs while the PWC-Net model was trained for 26 epochs. I found that using learning rate annealing was useful, since the loss tended to stagnate.

![Loss plot]({{ '/assets/images/loss.png' | relative_url }}){: style="width: 100%;" class="center"}

Overall Method #1, using Gunnar-Farneback optical flow, achieved a lower evaluation loss (~12 MSE) compared to that of Method #2 that used PWC-Net to estimate optical flow (~20 MSE). This could solely be due to the fact that I was severely confused on how I could turn the output of PWC-Net into a 3 channel RGB representation.

<iframe width="640" height="480" src="https://www.youtube.com/embed/ef5jz3NAdp8" frameborder="0"></iframe>{: style="padding-top: 25px;" class="center"}
<div align="center">
  Speeds for demonstration video generated using Method #1
</div>

## Notes on Generalization
`The speed challenge states that an MSE of <10 is good. <5 is better. <3 is heart.` I can firmly state that the methods demonstrated in this post do not generalize given the sparse training data that was provided. When I attempt to use the trained models to predict speed on the [third video](https://github.com/antoninodimaggio/Voof/blob/master/data/test/test.mp4) I get an MSE between ~70 and ~150. This is most likely due to the fact that the models were only trained on 16,420 frames. The model has simply not seen enough scenarios to generalize. At the start of the [demonstration video](https://www.youtube.com/embed/ef5jz3NAdp8) the car drives under an underpass, which in turn is followed by a large error spike. It is clear that the trained model has not seen the scenario in which a car drives under an under pass. However, the model has a super low error on long stretches of open highway, which is what the majority of the training video is composed of. The test video contains a lot of city driving which means a lot of intersections/stop and go traffic. The model was never trained on data like this so why would it be able to make an accurate prediction.

**Put simply, the models need to be trained on a lot more data to gain the ability to generalize.**

## Previous Work
* I found this to be the best resource: [https://github.com/ryanchesler/comma-speed-challenge](https://github.com/ryanchesler/comma-speed-challenge)
* [https://github.com/jovsa/speed-challenge-2017](https://github.com/jovsa/speed-challenge-2017)
* [https://github.com/JonathanCMitchell/speedChallenge](https://github.com/JonathanCMitchell/speedChallenge)

## Conclusion
I may expand on this work in the future. There are more [robust data sets](https://github.com/commaai/comma2k19){: rel="nofollow"} that I would like to explore. Thanks for reading!
