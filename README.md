# Modularity in neural networks

The [modularisation_via_noise](./modularisation_via_noise) folder contains basic scripts for training a deep non-linear network under various regularisers and noises, and assessing modular connectivity structure using the random walk Laplacian matrix. The python notebooks runs the training with various settings, and models and metrics are stored in the results folders.

The [modular_generalisation_Gaussian](./modular_generalisation_Gaussian) folder demonstrates the architecture constraint of a modularised network enables it to generalise beyond the training data domain. The [python notebook](./modular_generalisation_Gaussian/run.ipynb) trains three different network architectures to learn a 2-dimensional Gaussian, and only the modular one succeeds with partial data.

The [modular_generalisation_Random](./modular_generalisation_Random) folder is similar to the Gaussian variant, see [python notebook](./modular_generalisation_Random/run.ipynb) therein for particulars.

The [mod_bulk_competition](./mod_bulk_competition) folder shows how a modular block preferentially encodes non-linear transformations when connected to a fully connected block in series, and how this behaviour changes depending on the depths of the blocks. The [python notebook](./mod_bulk_competition/run.ipynb) compares the performances with a fixed number of fully connected layers and varying number of modular layers.

The [CIFAR10_modular](./CIFAR10_modular) folder extends the modularity investigation beyond synthetic tasks to a "film-roll" CIFAR-10 problem: each input is a horizontal concatenation of N independent CIFAR-10 frames and the target is the corresponding N-tuple of labels. [data.py](./CIFAR10_modular/data.py) provides a `ConcatCIFAR10` dataset that wraps `torchvision.datasets.CIFAR10` and stitches N frames into a `(3, 32, 32*N)` tensor with a length-N label vector. [model.py](./CIFAR10_modular/model.py) provides `ModularResNet18`, a CIFAR-style ResNet-18 whose final pooling preserves N positions along the width and whose fully connected layer is scaled to `(512*N) -> (10*N)` to emit one logit vector per frame. Training is not included yet.

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.