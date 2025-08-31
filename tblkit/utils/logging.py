import logging
from typing import Optional

_FMT = "[%(levelname)s] %(message)s"

def configure(level: int = logging.WARNING, *, quiet: bool = False,
              debug: bool = False, log_file: Optional[str] = None) -> None:
    if quiet:
        level = logging.ERROR
    if debug:
        level = logging.DEBUG
    handlers = [logging.StreamHandler()]
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        handlers.append(fh)
    logging.basicConfig(level=level, format=_FMT, handlers=handlers, force=True)

def get_logger(name: str = "tblkit"):
    return logging.getLogger(name)
