# ICGui ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=PyTorch&logoColor=white)&nbsp;![CUDA](https://img.shields.io/badge/-CUDA-76B900?logo=NVIDIA&logoColor=white)&nbsp;[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

Graphical User Interface for the NeRFICG framework

![Gui Demonstration GIF](resources/gui_demonstration.gif)

---

## Getting Started
Ensure NeRFICG has been installed and the appropriate conda environment is activated.

## Example Usage

### Standalone Viewer

The standalone viewer renders a pre-trained model from a checkpoint file.
By running the command below, a launcher is displayed to choose the 
model directory (or training config and checkpoint separately) and 
configure the GUI.
```shell
python ./scripts/gui.py  # from the base directory of the NeRFICG repository
```

The model to render as well as any GUI settings can also be passed on the command line; for more info run 
```shell
python ./scripts/gui.py --help
```

### Training Viewer

The model can also be viewed during training. This is done by simply enabling
the GUI in the config file before launching `scripts/train.py`. The GUI will start automatically when
training begins. Alternatively, you can also enable the GUI using

```shell
python ./scripts/train.py -c ./configs/config.yaml TRAINING.GUI.ACTIVATE=True
```

Depending on the frequency of rendering set in the config, this may slow down
training! The GUI can be closed at any point during training without affecting
the training process to free up resources used by the GUI.

## Acknowledgments

Special thanks go out to the following projects:

- [torchwindow](https://github.com/jbaron34/torchwindow), for demonstrating how to directly render tensors without copying them to the CPU.
