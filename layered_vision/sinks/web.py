import logging
import threading
from typing import Iterable, Optional
import cv2

import werkzeug.serving
from flask import Flask, Blueprint
from flask.helpers import url_for
from flask.wrappers import Response

from layered_vision.utils.image import rgb_to_bgr, apply_alpha
from layered_vision.utils.image import ImageArray

from .api import T_OutputSink


LOGGER = logging.getLogger(__name__)


class ServerThread(threading.Thread):
    def __init__(self, app: Flask, host: str, port: int, **kwargs):
        threading.Thread.__init__(self)
        self.server = werkzeug.serving.make_server(host, port, app, **kwargs)
        self.app_context = app.app_context()
        self.app_context.push()

    def run(self):
        LOGGER.info('starting server, host=%r, port=%d', self.server.host, self.server.port)
        self.server.serve_forever()

    def _shutdown(self):
        LOGGER.info('stopping webserver')
        self.server.shutdown()
        LOGGER.info('stopped webserver')

    def shutdown(self):
        threading.Thread(target=self._shutdown).start()


class LastImageFrameWrapper:
    def __init__(self):
        self.stopped_event = threading.Event()
        self.has_frame_event = threading.Event()
        self.frame: Optional[ImageArray] = None

    def stop(self):
        LOGGER.info('setting stopped event')
        self.stopped_event.set()

    def push(self, frame: ImageArray):
        self.frame = frame
        self.has_frame_event.set()

    def wait_for_next(self) -> ImageArray:
        while True:
            if self.stopped_event.is_set():
                LOGGER.info('stopped event set, raising StopIteration')
                raise StopIteration()
            if not self.has_frame_event.wait(1.0):
                continue
            self.has_frame_event.clear()
            frame = self.frame
            assert frame is not None
            return frame


def generate_image_frames(
    last_image_frame_wrapper: LastImageFrameWrapper
) -> Iterable[bytes]:
    while True:
        try:
            LOGGER.debug('waiting for frame...')
            frame = last_image_frame_wrapper.wait_for_next()
        except StopIteration:
            LOGGER.info('received stop iteration')
            return
        LOGGER.debug('generating frame...')
        bgr_frame = rgb_to_bgr(apply_alpha(frame.astype('float32')))
        (flag, encoded_frame) = cv2.imencode(".jpg", bgr_frame)
        if not flag:
            LOGGER.warning('unable to encode frame')
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_frame) + b'\r\n'
        )


class ServerBlueprint(Blueprint):
    def __init__(self, last_image_frame_wrapper: LastImageFrameWrapper):
        super().__init__('server', __name__)
        self.last_image_frame_wrapper = last_image_frame_wrapper
        self.route('/', methods=['GET'])(self.home)
        self.route('/stream', methods=['GET'])(self.stream)

    def home(self):
        return '<a href="%s">image stream</a>' % url_for('.stream')

    def stream(self):
        return Response(
            generate_image_frames(self.last_image_frame_wrapper),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )


class WebPreviewSink:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.last_image_frame_wrapper = LastImageFrameWrapper()
        self.app = Flask(__name__)
        self.app.register_blueprint(
            ServerBlueprint(self.last_image_frame_wrapper),
            url_prefix='/'
        )
        self.server_thread: Optional[ServerThread] = None

    def __enter__(self):
        self.server_thread = ServerThread(self.app, host=self.host, port=self.port)
        self.server_thread.start()
        return self

    def __exit__(self, *_, **__):
        self.last_image_frame_wrapper.stop()
        self.server_thread.shutdown()

    def __call__(self, image_array: ImageArray):
        self.last_image_frame_wrapper.push(image_array)


def get_web_preview_output_sink(
    *_, host='0.0.0.0', port: int = 8280, **__
) -> T_OutputSink:
    return WebPreviewSink(host=host, port=port)


OUTPUT_SINK_FACTORY = get_web_preview_output_sink
