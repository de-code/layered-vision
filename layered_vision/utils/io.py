import logging
import os
from hashlib import md5
from pathlib import Path
from urllib.request import urlopen


LOGGER = logging.getLogger(__name__)


def is_remote_file_path(file_path: str) -> bool:
    return '://' in str(file_path)


def get_file_to(
    remote_file_path: str,
    relative_local_file_path: str
) -> str:
    if os.path.exists(remote_file_path):
        LOGGER.debug('file is local file: %s', remote_file_path)
        return remote_file_path
    local_file = os.path.join('data', '.cache', relative_local_file_path)
    if os.path.exists(local_file):
        LOGGER.debug('file already downloaded: %s', local_file)
        return local_file
    with urlopen(remote_file_path) as response:
        data = response.read()
    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    with open(local_file, mode='wb') as fp:
        fp.write(data)
    return local_file


def strip_url_suffix(path: str) -> str:
    qs_index = path.find('?')
    if qs_index > 0:
        return path[:qs_index]
    return path


def get_file(file_path: str, download: bool = True) -> str:
    if not download:
        return file_path
    if os.path.exists(file_path):
        return file_path
    local_path = get_file_to(
        file_path,
        (
            md5(file_path.encode('utf-8')).hexdigest()
            + '-'
            + os.path.basename(strip_url_suffix(file_path))
        )
    )
    return local_path


def read_text(file_path: str, encoding: str = 'utf-8') -> str:
    if not is_remote_file_path(file_path):
        return Path(file_path).read_text(encoding=encoding)
    with urlopen(file_path) as response:
        return response.read().decode(encoding=encoding)
