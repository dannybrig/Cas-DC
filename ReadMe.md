For a more detailed technical description, please read our ICIP 2024 paper "Cascading Unknown Detection with Known Classification for Open Set Recognition."

## Models folder
    - This folder contains the models for training. One for classication and the other for learning embeddgings.

        - classifier_resnet.py contains a ResNet implemenation for classification of knowns
        - embed_resnet.py contains a ResNet implenataion for the embedding network

        - Note: The difference between classifier and embedding networks lies in the final activation of the logits.

## utils.py
    - This file contains various utilities used throughout the OSR process.
        - Extract embedding network embeddings from entire dataset
        - Calculate distance to each of the known and unknown prototypes (means in embedding space)
        - Make known/unknown classification by thresholding the distance to the unknown prototype
        - Obtain confusion matrix for known+1 (all knowns plus remaining unknown class)
        - Obtain known/unknown prediction based on distance (NOT USED)

## data_generator.py
    - This file is used to generated the known knowns, known unknowns, and unknown unknowns for each dataset. When generating known knowns data, a mapping takes place to condense the output space to total number of knowns (e.g., if known classes are [0,2,4,6,8,9], the labels then get mapped to [0,1,2,3,4,5])
    - Supported datasets: MNIST, SVHN, CIFAR10, CIFAR+10, CIFAR+100, Tiny Imagenet

## tiny_imgnet.py
    - Contains class to support dataloading of Tiny Imagenet dataset

## main.py
    - Main function for experimentation.
    - Args:
        - dataset: which dataset to experiment on
        - known_data_dir: path to known knowns data
        - unknown_data_dir: path to unknown data
        - beta: margin to use in triplet loss function
        - ku_percentage: percentage of unknown to use as known unknowns
        - in_channels: number of input channels to network
        - batch_size: batch size
        - classifier_epochs: Number of epochs to train classifier network
        - classifier lr: learning rate for classifier network
        - embednet_epochs: number of epochs to train embedding network
        - embednet_lr: learninig rate for embedding network
        - num_trials: total number of experimental trials to perform on dataset

    - Order of operations
        1. Load data
        2. Train classifier
        3. Obtain classifier predictions
        4. Train Embedding network
        5. Get embeddings of training data
        6. Find mean of known knowns and known unknowns embedding (to be used as prototypes)
        7. Obtain test data embeddings
        8. Find distance of test data embeddings to each prototype
        9. Threshold by distance to unknown prototype
        10. Make predictions