import re
from typing import Optional, Tuple


def parse_type_path(path: str) -> Tuple[Optional[str], str]:
    m = re.match(r'^([a-z]+)(:(([^/]|/[^/]).*|))?$', path)
    if m:
        return m.group(1), m.group(3) or ''
    return None, path
