# MedNCA
PyTorch implementation of a Neural Cellular Automaton as proposed by the Paper ["Growing Neural Cellular Automata"](https://distill.pub/2020/growing-ca/), trained on medical images from [MedMNIST](https://medmnist.com/) and visualized using Pygame.

This project was done as part of the lecture Deep Generative Models at [TU Darmstadt](https://www.tu-darmstadt.de/index.en.jsp).

| ![GIF 1](demo/frames/videos/mednca_blood0_800steps.gif) | ![GIF 2](demo/frames/videos/mednca_blood0_regeneration.gif) | ![GIF 3](demo/frames/videos/mednca_retina0_200steps.gif) | ![GIF 4](demo/frames/videos/mednca_retina0_regeneration.gif) |
| ------------------------------------------------------- | ----------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------ |

## Dependencies

See `requirements.txt`.

## Usage

Run `main.py` to start the interactive demonstration (no GPU required).

Edit and run `schedule_training.py` or `start_training.py` to train a new model (GPU recommended).

## Contributors

- Lukas Maninger
- Fabian Metschies
- Patrick Siebke
- Patrick Vimr
