import os
from contextlib import closing
import socket


def find_free_port() -> int:
    """
    Find a free port for dist url
    :return:
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]

    return port
