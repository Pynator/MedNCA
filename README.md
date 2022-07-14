# MedNCA
PyTorch implementation of a Neural Cellular Automaton as proposed by the paper ["Growing Neural Cellular Automata"](https://distill.pub/2020/growing-ca/), trained on medical images from [MedMNIST](https://medmnist.com/) and visualized using Pygame.

This project was done as part of the lecture Deep Generative Models at [TU Darmstadt](https://www.tu-darmstadt.de/index.en.jsp).

| <img src="demo/frames/videos/mednca_blood0_800steps.gif" alt="GIF 1" width="140" height=auto /> | <img src="demo/frames/videos/mednca_blood0_regeneration.gif" alt="GIF 2" width="140" height=auto /> | <img src="demo/frames/videos/mednca_retina0_200steps.gif" alt="GIF 3" width="140" height=auto /> | <img src="demo/frames/videos/mednca_retina0_regeneration.gif" alt="GIF 4" width="140" height=auto /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

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
