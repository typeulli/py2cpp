from functools import wraps
from pathlib import Path
from typing import TypeVar, Callable, ParamSpec
import os
import sys
import time

T = TypeVar("T")
def assert_type(value: object, type_: type[T]) -> T:
    assert isinstance(value, type_), f"Expected type {type_.__name__}, got {value!r}"
    return value

P = ParamSpec("P")
def get_time(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} execution time: {end_time - start_time} seconds")
        return result
    return wrapper


def get_cache_dir() -> Path:
    """
    Return OS-standard cache directory for py2cpp.
    Windows:   %LOCALAPPDATA%/py2cpp
    Linux:     $XDG_CACHE_HOME/py2cpp  (default: ~/.cache/py2cpp)
    macOS:     ~/Library/Caches/py2cpp
    """
    if sys.platform.startswith("win"):
        # %LOCALAPPDATA%
        base_str = os.environ.get("LOCALAPPDATA")
        if base_str:
            base = Path(base_str)
        else:
            # fallback (very rare)
            base = Path.home() / "AppData" / "Local"
        

    elif sys.platform == "darwin":
        # ~/Library/Caches
        base = Path.home() / "Library" / "Caches"

    else:
        # Linux / Unix
        # XDG_CACHE_HOME or ~/.cache
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

    cache_dir = base / "py2cpp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir