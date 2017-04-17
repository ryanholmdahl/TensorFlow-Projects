# TensorFlow-Projects

This repository contains several TensorFlow projects, each of which implement cutting-edge deep learning research.
1. [Adversarial Autoencoders.](https://arxiv.org/pdf/1511.05644v2.pdf) Like variational autoencoders, adversarial autoencoders learn a compressed representation of the input that is forced into the shape of a given probability distribution. At each iteration, an autoencoder is trained. Its style outputs are then fed into a discriminator network, which learns to differentiate them from samples pulled from the desired distribution. The encoder network is then updated to better confuse the discriminator. Over time, this creates a fully functional autoencoder whose hidden representations are indistinguishable from samples from the desired distribution. This allows one to pull a new sample from that distribution, input it to the decoder network, then receive with high probability a completely new but sensible output.  
  
    In this project, I used the adversarial autoencoder on the MNIST dataset. After training, a new Gaussian sample can be input to create digits of an entirely new handwriting style.
  
2. [Deep Ensemble Learning.](https://arxiv.org/pdf/1602.02285v1.pdf) Deep ensemble learning uses the predictions of several weak, possibly dependent, classifiers to make more accurate predictions. By nonlinearly altering the input, as a deep network is capable of doing, the dependencies between classifiers can be modeled and used to make more effective predictions than any single classifier or the majority vote.
  
    In this project, I created many linear classifiers on portions of the MAGIC dataset, then trained a deep ensemble learner on their outputs.
  
3. [Neural Document Modeling.](https://arxiv.org/pdf/1511.06038v4.pdf) The paper describes a Neural Varitaional Document Model, which maps a continuous bag-of-words (CBOW) representation of a document to a continuous style vector. I took a somewhat different approach from the linked paper, opting to not include the variational framework and instead adding regularization to the style representation to prevent value explosion. Regardless, the styles can still be used to analyze document similarity and compress information to a much lower dimension than the full vocabulary.
  
    In this project, I created a neural document model to find an effective reduction of the 20News dataset. I used [this work](https://github.com/carpedm20/variational-text-tensorflow) for a few implementation pointers, especially with building the loss function.
