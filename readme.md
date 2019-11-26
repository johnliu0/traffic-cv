# Traffic-cv
Three-signal traffic light recognition. The aim of this project is to build a fast and performant system for the recognition and localization of traffic lights. Currently, I'm in the phase of building a functional R-CNN (as described in the paper by [Ross Girshick et al.](https://arxiv.org/abs/1311.2524)) that can localize traffic lights in a given image.

The ultimate goal is to develop an accurate system that can detect in real-time.

News, updates, and demos available at [johnliu.io/traffic-cv](https://johnliu.io/traffic-cv)

---

# Implementation
As mentioned above, an R-CNN is used for detection and localization. In the original paper, the components of an R-CNN were broken down and described in three parts. Here is a quick summary:

1. **Region proposal**: bounding boxes for potential objects are generated. In the original paper, selective search was used; and there exists other region proposal algorithms as well. In traffic-cv, a modified selective search is implemented and used.

2. **CNN feature extraction**: the region proposals are then fed into a convnet for feature extraction. The VGG16 convnet was used as a base for traffic-cv. Using a pretrained convnet greatly helps with detection accuracy since a lot of functionality from convnets are easily generalizable.

3. **SVM classification**: the convnet output is finally fed into SVMs for classification; one SVM per class. Similarly classified bounding boxes can be merged if their intersection-over-union exceeds a certain threshold.

I use Scikit-image's `felzenszwalb` algorithm to initially segment an image and then implement a modified selective search based on the paper by [J.R.R. Uijlings et. al](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf).

The VGG16 base pretrained on the ImageNet database is used for feature extraction.

Finally, one linear SVM is used to classify whether or not the convnet output corresponds to a traffic light. I train the SVM using my own private data that I have been collecting.

---

# File Structure

- `main.py` Entry point.
- `region.py` Proposes object regions.
- `convnet.py` CNN model used for feature extraction.
- `classifier.py` SVM model used for object classification.
- `augment.py` Provides data augmentation tools for training.

---

# Roadmap

I split the project develop into several major phases. Each phase introduces a major feature or change into this project. There are no strict timelines for these phases as its based on how much time I am allocate for this project.

## Phase 1

R-CNN development and developing a working prototype.

- [x] Data collection
- [x] Region proposal using selective search
- [x] Training an SVM to classify an image as a traffic light
- [ ] Optimizing the system for performance

## Phase 2

- [ ] Training multiple SVMs to recognize the different states of traffic lights (e.g. red light, yellow light, green light)
- [ ] Refine data and training process; consider low-light environments, different weather, partial obstruction (by a tree, for example); fine-tuning

## Phase 3

- [ ] Look into fast video processing techniques and implement real-time detection (multiple recognitions per second)

## Phase 4

- [ ] Put service into cloud and build this app on mobile device (very dependent on the amount of disposable income I have this since the compute power I need will not come cheap)
