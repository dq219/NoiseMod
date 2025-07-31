# Modularity in neural networks

The [modularisation_via_noise](./modularisation_via_noise) folder contains basic scripts for training a deep non-linear network under various regularisers and noises, and assessing modular connectivity structure using the random walk Laplacian matrix. The [python notebook](./modularisation_via_noise/run.ipynb) runs the training with various settings, and models and metrics are stored in the [results folder](./modularisation_via_noise/results_nonlinear).

The [modular_generalisation_Gaussian](./modular_generalisation_Gaussian) folder demonstrates the architecture constraint of a modularised network enables it to generalise beyond the training data domain. The [python notebook](./modular_generalisation_Gaussian/run.ipynb) trains three different network architectures to learn a 2-dimensional Gaussian, and only the modular one succeeds with partial data.

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.