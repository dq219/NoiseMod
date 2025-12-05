# Modularity in neural networks

The [modularisation_via_noise](./modularisation_via_noise) folder contains basic scripts for training a deep non-linear network under various regularisers and noises, and assessing modular connectivity structure using the random walk Laplacian matrix. The [python notebook](./modularisation_via_noise/run.ipynb) runs the training with various settings, and models and metrics are stored in the [results folder](./modularisation_via_noise/results).

The [modular_generalisation_Gaussian](./modular_generalisation_Gaussian) folder demonstrates the architecture constraint of a modularised network enables it to generalise beyond the training data domain. The [python notebook](./modular_generalisation_Gaussian/run.ipynb) trains three different network architectures to learn a 2-dimensional Gaussian, and only the modular one succeeds with partial data.

The [modular_generalisation_Random](./modular_generalisation_Random) folder is similar to the Gaussian variant, see [python notebook](./modular_generalisation_Random/run.ipynb) therein for particulars.

The [mod_bulk_competition](./mod_bulk_competition) folder shows how a modular block preferentially encodes non-linear transformations when connected to a fully connected block in series, and how this behaviour changes depending on the depths of the blocks. The [python notebook](./mod_bulk_competition/run.ipynb) compares the performances with a fixed number of fully connected layers and varying number of modular layers.

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.