# Layered Vision

[![PyPi version](https://img.shields.io/pypi/v/layered-vision)](https://pypi.org/project/layered-vision/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Goals of this project is:

* A tool to allow the composition of images or videos via a configuration file (e.g. as a virtual webcam).

This project is still very much experimental and may change significantly.

## Install

Install with all dependencies:

```bash
pip install layered-vision[all]
```

Install with minimal dependencies:

```bash
pip install layered-vision
```

Extras are provided to make it easier to provide or exclude dependencies
when using this project as a library:

| extra name | description
| ---------- | -----------
| bodypix    | For [bodypix](https://github.com/de-code/python-tf-bodypix) filter
| webcam     | Virtual Webcam support via [pyfakewebcam](https://pypi.org/project/pyfakewebcam/)
| youtube    | YouTube support via [pafy](https://pypi.org/project/pafy/) and [youtube_dl](https://pypi.org/project/youtube_dl/)
| mediapipe  | Selfie Segmentation using [MediaPipe](https://google.github.io/mediapipe/solutions/selfie_segmentation.html).
| all        | All of the libraries

## Virtual Webcam For Linux

You do not need to use a webcam to use the project, as you could feed a video file.
But if you do want to use a webcam (currently only supported on Linux), this section provides a bit more information.

On Linux, `/dev/video0` often refers to the true webcam device.

You can use [v4l2loopback](https://github.com/umlaeute/v4l2loopback)
to create a virtual webcam device. e.g. you could `/dev/video2`.

Most applications looking for a webcam should then be able to use that virtual device.
(Applications might include Chromium, Skype etc.)

Once installed, you can create `/dev/video2` via the following command:

```bash
modprobe v4l2loopback devices=1 video_nr=2 exclusive_caps=1 card_label="VirtualCam 1"
```

To create the device after every reboot, you might want to create `/etc/modprobe.d/v4l2loopback.conf`:

```text
options v4l2loopback devices=1 video_nr=2 exclusive_caps=1 card_label="VirtualCam 1"
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
  input_path: "https://raw.githubusercontent.com/numpy/numpy/v1.20.1/branding/logo/logomark/numpylogoicon.png"
- id: out
  output_path: "numpy-logo.png"
```

You could also have two outputs (`example-config/two-outputs.yml`):

```yaml
layers:
  - id: in
    input_path: "https://raw.githubusercontent.com/numpy/numpy/v1.20.1/branding/logo/logomark/numpylogoicon.png"
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
    input_path: "https://raw.githubusercontent.com/numpy/numpy/v1.20.1/branding/logo/logomark/numpylogoicon.png"
    width: 480
    height: 300
    repeat: true
  - id: out
    output_path: window
```

### Input Layer (Source)

A layer that has an `input_path` property.

The following inputs are currently supported:

| type name | description |
| -----| ----------- |
| image | Static image (e.g. `.png`) |
| video | Video (e.g. `.mp4`) |
| webcam | Linux Webcam (`/dev/videoN`) |
| fill | Fills a new image with a color |
| youtube | YouTube stream (e.g. `https://youtu.be/f0cGgOv3l4c`,  see [example config](https://github.com/de-code/layered-vision/tree/develop/example-config/display-video-bodypix-replace-background-youtube.yml)) |
| mss | Screen capture using [mss](https://python-mss.readthedocs.io/index.html) (see [example config](https://github.com/de-code/layered-vision/tree/develop/example-config/display-video-bodypix-replace-background-mss.yml)) |

The `input_path` may point to a remote location (as is the case with [all examples](https://github.com/de-code/layered-vision/tree/develop/example-config)). In that case it will be downloaded and cached locally.

In most cases the *type name* is inferred from the `input_path`.
You can also specify the type explicitly via the `type` property or by prefixing the path, e.g.: `webcam:/dev/video0`.

### Output Layer (Sink)

A layer that has an `output_path` property.

The following outputs are currently supported:

| type name | description |
| -----| ----------- |
| image_writer | Write to a static image (e.g. `.png`) |
| v4l2 | Linux Virtual Webcam (`/dev/videoN`) |
| window | Display a window |
| web | Provide output as JPEG stream |

As is the case with the `input_path`, in most cases the *type name* is inferred from the `output_path`.
You can also specify the type explicitly via the `type` property or by prefixing the path, e.g.: `v4l2:/dev/video2`.

#### Web Stream (Experimental)

The output may also be provide as a JPEG stream. That way it can be viewed in a browser.

The following configuration options are supported:

| name | default value | description |
| ---- | ------------- | ----------- |
| `host` | `0.0.0.0`   | Host to listen to, `0.0.0.0` to listen on any host. This could also be set to say `127.0.0.1` to prevent the stream from being accessed from another machine.
| `port` | `8280`      | The port to listen to.

With the default configuration, opening `http://localhost:8280` will provide a link to the stream.
The stream will contineously provide JPEG frames to the browser (as a single request).

Currently only one stream consumer is supported.

### Filter Layer

A layer that has a `filter` property.

The following filters are currently supported:

| name | description |
| -----| ----------- |
| `box_blur` | Blurs the image or channel. |
| `bodypix` | Uses the [bodypix](https://github.com/de-code/python-tf-bodypix) model to mask a person. |
| `chroma_key` | Uses a chroma key (colour) to add a mask |
| `copy` | Copies the input. Mainly useful as a placeholder layer with `branches`. |
| `dilate` | Dilates the image or channel. For example to increase the alpha mask after using `erode` |
| `erode` | Erodes the image or channel. That could be useful to remove outliers from an alpha mask. |
| `bilateral` | Applies a [bilateral filter](https://en.wikipedia.org/wiki/Bilateral_filter), using `d`, `sigma_color` and `sigma_space` parameters. |
| `motion_blur` | Adds a motion blur to the image or channel. That could be used to make an alpha mask move more slowly |
| `mp_selfie_segmentation` | Uses the [MediaPipe's Selfie Segmentation](https://google.github.io/mediapipe/solutions/selfie_segmentation.html) to mask a person (similar to bodypix). |
| `pixelate` | Pixelates the input. |
| `fill` | Fills the input or a selected channel with a color / value. e.g. with `color` set to `blue` |
| `invert` | Inverts the input. e.g. `black` to `white` |
| `multiply` | Multiplies the input with a constant value. e.g. to adjust the `alpha` channel |
| `warp_perspective` | Warps the perspective of the input image given a list of `target_points`. e.g. to display it in a corner of the output image |

Every *filter* may have additional properties. Please refer to the [examples](https://github.com/de-code/layered-vision/tree/develop/example-config) (or come back in the future) for more detailed information. In particular [display-video-segmentation-replace-background-template.yml](https://github.com/de-code/layered-vision/blob/develop/example-config/display-video-segmentation-replace-background-template.yml) provides examples of most filters (often disabled by default).

#### Filter: mp_selfie_segmentation (Experimental)

[MediaPipe's Selfie Segmentation](https://google.github.io/mediapipe/solutions/selfie_segmentation.html) allows background segmentation (similar to bodypix). As it is more optimized, it will usually be faster than using bodypix.

The following parameters are supported:

| name | default value | description |
| ---- | ------------- | ----------- |
| `model_selection` | `1`   | The model to use, `0` or `1` (please refer [MediaPipe's Selfie Segmentation documentation](https://google.github.io/mediapipe/solutions/selfie_segmentation.html#model_selection) for further details).
| `threshold` | `0.1`      | The threshold for the segmentation mask.
| `cache_model_result_secs` | `0.0`      | The number of seconds to cache the mask for.

```bash
python -m layered_vision start \
  --config-file \
  "example-config/display-video-segmentation-replace-background-template.yml" \
  --set "bodypix.enabled=false" \
  --set "mp_selfie_segmentation.enabled=true"
```

### Branches Layer

A layer that has an `branches` property.
Each *branch* is required to have a `layers` property.
The input to each set of *branch layers* is the input to the *branches* layer.
The *branches* are then combined (added on top of each other).
To make *branches* useful, at least the last *branch image* should have an alpha mask.

### Error Handling

By default, any error such as an invalid path or filter parameter,
will result in an exception being thrown, causing the application to exit.

To instead display an image, you could define an input layer with the id `on_error`:

```yaml
layers:
  - id: on_error
    # Source: https://pixabay.com/vectors/test-pattern-tv-tv-test-pattern-152459/
    input_path: "https://www.dropbox.com/s/29ycjg9ubht776y/test-pattern-152459_640.png?dl=1"
    repeat: true
  # ...
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

You could also load the config from a remote location:

```bash
python -m layered_vision start --config-file \
  "https://raw.githubusercontent.com/de-code/layered-vision/develop/example-config/display-video-chroma-key-replace-background.yml"
```

It is also possible to override config values via command line arguments, e.g.:

```bash
python -m layered_vision start --config-file=example-config/display-image.yml \
    --set out.output_path=/path/to/output.png
```

You could also try replacing the background with a YouTube stream:

```bash
python -m layered_vision start \
  --config-file \
  "https://raw.githubusercontent.com/de-code/layered-vision/develop/example-config/webcam-bodypix-replace-background-to-v4l2loopback.yml" \
  --set bg.input_path="https://youtu.be/f0cGgOv3l4c" \
  --set bg.fps=30 \
  --set in.input_path="/dev/video0" \
  --set out.output_path="/dev/video2"
```

Note: you may need to specify the fps

If a local configuration file was specified, the application will attempt to reload it on change.

### Docker Usage

You could also use the Docker image if you prefer.
The entrypoint will by default delegate to the CLI, except for `python` or `bash` commands.

```bash
docker pull de4code/layered-vision

docker run --rm \
  --device /dev/video0 \
  --device /dev/video2 \
  de4code/layered-vision start \
  --config-file \
  "https://raw.githubusercontent.com/de-code/layered-vision/develop/example-config/webcam-bodypix-replace-background-to-v4l2loopback.yml" \
  --set bg.input_path="https://www.dropbox.com/s/4debg4lrgn5g36l/toy-train-3288425.mp4?dl=1" \
  --set in.input_path="/dev/video0" \
  --set out.output_path="/dev/video2"
```

(Background: [Toy Train](https://www.pexels.com/video/toy-train-3288425/))

## Acknowledgements

* [virtual_webcam_background](https://github.com/allo-/virtual_webcam_background), a somewhat similar project (more focused on bodypix)
* [OBS Studio](https://obsproject.com/), conceptually a source of inspiration. (with UI etc)
