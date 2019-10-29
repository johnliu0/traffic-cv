# traffic-cv
Three-signal traffic light recognition. This project focuses around traffic lights in Toronto.



# Implementation
The R-CNN methodology as described by Ross Girshick (and others) is used here. This algorithm for fast object detection and localization revolves around three steps:

1. **Region proposal**: bounding boxes for potential objects are generated. In the original paper, selective search was used; and there exists other region proposal algorithms as well. In traffic-cv, a custom algorithm with Felzenszwalb's image segmentation method is used.

2. **CNN feature extraction**: the region proposals are then fed into a convnet for feature extraction. The <insert convnet> convnet was used as a base for traffic-cv. Using a pretrained convnet greatly helps with detection accuracy since a lot of features from convnets are easily generalizable.

3. **SVM classification**: the convnet output is finalled fed into numerous SVMs for classification. One SVM is required for each class. Each SVM can then give a score as to how likely a region is to be of a particular class.


# Roadmap
___

I split the project develop into several major phases. Each phase introduces a major feature or change into this project. There are no strict timelines for these phases as its based on how much time I am allocate for this project.

## Phase 1

Getting out a minimally viable but structurally sound product here is important. The goal here is to work on a basic system that can detect and identify traffic lights in a given image. I implement Ross Girshick's (and others) R-CNN method here.

- [ ] Data collection (at least a few hundred images of traffic lights)
- [ ] Region proposals using Felzenszwalb's image segmentation algorithm
- [ ] Build upon a pretrained convnet
- [ ] Training a single SVM to recognize whether or not a traffic light exists in an image (binary classification)

## Phase 2

- [ ] Training multiple SVMs to recognize the different states of traffic lights (e.g. red light, yellow light, green light)

## Phase 3

- [ ] Look into fast video processing techniques and attempting to implement real-time detection

## Phase 4

- [ ] Put service into cloud and build this app on mobile device (very dependent on the amount of disposable income I have this since the compute power I need will not come cheap)
