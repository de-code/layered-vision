# Layered Vision

[![PyPi version](https://pypip.in/v/layered-vision/badge.png)](https://pypi.org/project/layered-vision/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Goals of this project is:

* A tool to allow the composition of images or videos via a configuration file (e.g. as a virtual webcam).

This project is still very much experimental and may change significantly.

## Install

```bash
pip install layered-vision
```

## Configuration

The configuration format is file is [YAML](https://en.wikipedia.org/wiki/YAML).

There are a number of [example configuration files](https://github.com/de-code/layered-vision/tree/develop/example-config).

### Layers

Every configuration file will contain layers. Layers are generally described from top to down.
With the last layer usually being the output layer.

The source to the output layer will be the layer above.

A very simple configuration file that downloads the `numpy` logo and saves it to a file might look like (`example-config/save-image.yml`):

```yaml
layers:
- id: in
  input_path: "https://github.com/numpy/numpy/raw/master/branding/logo/logomark/numpylogoicon.png"
- id: out
  output_path: "numpy-logo.png"
```

You could also have two outputs (`example-config/two-outputs.yml`):

```yaml
layers:
  - id: in
    input_path: "https://github.com/numpy/numpy/raw/master/branding/logo/logomark/numpylogoicon.png"
  - id: out1
    output_path: "data/numpy-logo1.png"
  - id: out2
    output_path: "data/numpy-logo2.png"
```

In that case the source layer for both `out1` and `out2` is `in`.

By using `window` as the `output_path`, the image is displayed in a window (`example-config/display-image.yml`):

```yaml
layers:
  - id: in
    input_path: "https://github.com/numpy/numpy/raw/master/branding/logo/logomark/numpylogoicon.png"
    width: 480
    height: 300
    repeat: true
  - id: out
    output_path: window
```

### Input Layer

A layer that has an `input_path` property.

The following inputs are currently supported:

* Image
* Video
* Linux Webcam (`/dev/videoN`)

The `input_path` may point to a remote location (as is the case with [all examples](https://github.com/de-code/layered-vision/tree/develop/example-config)). In that case it will be downloaded and cached locally.

### Filter Layer

A layer that has an `filter` property.

The following filters are currently supported:

| name | description |
| -----| ----------- |
| `box_blur` | Blurs the image or channel. |
| `bodypix` | Uses the [bodypix](https://github.com/de-code/python-tf-bodypix) model to mask a person. |
| `chroma_key` | Uses a chroma key (colour) to add a mask |
| `copy` | Copies the input. Mainly useful as a placeholder layer with `branches`. |
| `dilate` | Dilates the image or channel. For example to increase the alpha mask after using `erode` |
| `erode` | Erodes the image or channel. That could be useful to remove outliers from an alpha mask. |
| `motion_blur` | Adds a motion blur to the image or channel. That could be used to make an alpha mask move more slowly |
| `pixelate` | Pixelates the input. |

Every *filter* may have additional properties. Please refer to the [examples](https://github.com/de-code/layered-vision/tree/develop/example-config) (or come back in the future) for more detailed information.

### Branches Layer

A layer that has an `branches` property.
Each *branch* is required to have a `layers` property.
The input to each set of *branch layers* is the input to the *branches* layer.
The *branches* are then combined (added on top of each other).
To make *branches* useful, at least the last *branch image* should have an alpha mask.

## CLI

### CLI Help

```bash
python -m layered_vision --help
```

or

```bash
python -m layered_vision <sub command> --help
```

### Example Command

```bash
python -m layered_vision start --config-file=example-config/display-image.yml
```

You could also load the config from a remote location:

```bash
python -m layered_vision start --config-file \
  "https://raw.githubusercontent.com/de-code/layered-vision/develop/example-config/display-video-chroma-key-replace-background.yml"
```

## Acknowledgements

* [virtual_webcam_background](https://github.com/allo-/virtual_webcam_background), a somewhat similar project (more focused on bodypix)
* [OBS Studio](https://obsproject.com/), conceptually a source of inspiration. (with UI etc)
