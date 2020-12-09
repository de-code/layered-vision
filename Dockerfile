FROM python:3.8-buster

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        dumb-init libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/layered-vision

COPY requirements.build.txt ./
RUN pip install --disable-pip-version-check --user -r requirements.build.txt

COPY requirements.txt ./
RUN pip install --disable-pip-version-check --user -r requirements.txt

COPY layered_vision ./layered_vision
COPY example-config ./example-config

COPY docker/entrypoint.sh ./docker/entrypoint.sh

ENTRYPOINT ["/usr/bin/dumb-init", "--", "/opt/layered-vision/docker/entrypoint.sh"]
