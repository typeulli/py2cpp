import platform

from py2cpp.utils.code import get_cache_dir

SYSTEM_NAME = platform.system().lower()

PATH_COMPILER = "g++"
PATH_COMPILER_PYD = "cl"

PATH_CACHE = get_cache_dir()
PATH_CACHE_JIT = PATH_CACHE / "jit"