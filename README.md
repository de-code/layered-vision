# Layered Vision

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Install

TODO

## Configuration

The configuration format is file is [YAML](https://en.wikipedia.org/wiki/YAML).

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
