import platform
import sysconfig

from pathlib import Path

from py2cpp.utils.code import get_cache_dir

SYSTEM_NAME = platform.system().lower()

PATH_COMPILER = "g++"
PATH_COMPILER_PYD = "cl"

PATH_CACHE = get_cache_dir()
PATH_CACHE_JIT = PATH_CACHE / "jit"

PATH_CPYTHON_BASE = Path(sysconfig.get_config_var("installed_base"))
PATH_CPYTHON_INCLUDE = Path(sysconfig.get_config_var("INCLUDEPY"))
PATH_CPYTHON_LIB = PATH_CPYTHON_BASE / "libs"