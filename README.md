# ICGUI
Graphical User Interface for the NeRFICG framework

![Python](https://img.shields.io/static/v1?label=Python&message=3&color=success&logo=Python)&nbsp;![OS](https://img.shields.io/static/v1?label=OS&message=Linux/macOS&color=success&logo=Linux)&nbsp;[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

![Gui Demonstration GIF](resources/gui_demonstration.gif)

---

## Getting Started
Ensure NeRFICG has been installed (do not specify -h / --headless, as this installs the project without 
GUI dependencies) and the appropriate conda environment is activated.

Optionally install matplotlib for system font support:
```sh
pip install matplotlib
```

## Example Usage

### Standalone Viewer

The standalone viewer renders a pre-trained model from a checkpoint file.
By running the command below, a launcher is displayed to choose the 
model directory (or training config and checkpoint separately) and 
configure the GUI.
```sh
./scripts/gui.py  # from the base directory of the NeRFICG repository
./launchViewer.py  # from the ICGui directory
```

The model to render as well as any GUI settings can also be passed on the command line; for more info run 
```sh
./scripts/gui.py --help
```

### Training Viewer

The model can also be viewed during training. This is done by simply enabling
the GUI in the config file. The GUI will then be launched automatically when
training is started. From the base directory of the NeRFICG repository, run

```sh
./scripts/train.py --config ./configs/config.yaml
```

Depending on the frequency of rendering set in the config, this may slow down
training! The GUI can be closed at any point during training without affecting
the training process to free up resources used by the GUI.

## Acknowledgments

Special thanks go out to the following projects:

- [torchwindow](https://github.com/jbaron34/torchwindow), for demonstrating how
to directly render tensors without copying them to the CPU.
