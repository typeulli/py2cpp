
from ctypes import wintypes, windll, CDLL
import ctypes
import hashlib
import inspect
import platform
import subprocess

from functools import wraps
from pathlib import Path
from typing import Callable, Generic, ParamSpec, TypeVar, overload

from py2cpp.core.compiler import Setting, py_2_cpp, FunctionTypeData, parse_type
from py2cpp.utils.code import get_cache_dir

system_name = platform.system().lower()

PATH_COMPILER = "g++"
PATH_CACHE = get_cache_dir() / "jit"
P = ParamSpec("P")
T = TypeVar("T")

class JitFunction(Generic[P, T]):
    
    @classmethod
    def hash(cls, func: Callable[P, T], cxx: int, o: int) -> str:
        hasher = hashlib.sha256()
        hasher.update(inspect.getsource(func).encode("utf-8"))
        hasher.update(f"{cxx}:{o}".encode("utf-8"))
        return hasher.hexdigest()

    def __init__(self, func: Callable[P, T], path_clib: str, type_ret: type, argtypes: list[type]):
        self.lib = CDLL(path_clib)
        self.handle = self.lib._handle
        self.path = path_clib
        
        self.func: Callable[P, T] = func
        self.cfunc = getattr(self.lib, func.__name__)
        self.cfunc.restype = type_ret
        self.cfunc.argtypes = argtypes
        wraps(func)(self)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.cfunc(*args)

    def unload(self):
        if system_name == "windows":
            if hasattr(self, "handle"):
                windll.kernel32.FreeLibrary(wintypes.HMODULE(self.handle))
                del self.handle

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass
        try:
            Path(self.path).unlink(missing_ok=True)
        except Exception:
            pass

    def __hash__(self) -> int:
        return self.hash(self.func, 17, 3).__hash__()

@overload
def jit(func: Callable[P, T], *, cxx: int = 17, o: int = 3) -> Callable[P, T]: ...
@overload
def jit(func: None = None, *, cxx: int = 17, o: int = 3) -> Callable[[Callable[P, T]], Callable[P, T]]: ...
def jit(func: Callable[P, T] | None = None, *, cxx: int = 17, o: int = 3) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    def wrapper(func: Callable[P, T]) -> Callable[P, T]:
        source = inspect.getsource(func)
        assert source.startswith("@")
        source = source[source.index("\n")+1:]
        cpp_data = py_2_cpp(source, path=f"<jit:{func.__name__}>", setting=Setting())
        cpp_code = cpp_data.code
        
        
        _type_fn = cpp_data.type_ctx.get_vartype(f"{func.__name__}")
        assert type(_type_fn) == FunctionTypeData
        type_fn: FunctionTypeData = _type_fn
        type_ret = parse_type(type_fn.return_type, cpp_data.type_ctx, cpp_data.state)
        fn_code = type_ret + " " + func.__name__ + "("
        for i, arg in enumerate(type_fn.args):
            arg_type = parse_type(arg[1], cpp_data.type_ctx, cpp_data.state)
            fn_code += arg_type + " " + arg[0]
            if i != len(type_fn.args) - 1:
                fn_code += ", "
        fn_code += ")"
        assert fn_code in cpp_code
        cpp_code = cpp_code.replace(fn_code, "extern \"C\" __declspec(dllexport) " + fn_code)
        cpp_code = cpp_code[:cpp_code.rindex("int main()")].strip()


        executing_path = Path(inspect.getfile(func))
        pycache_path = executing_path.parent / "__pycache__"
        output = pycache_path / (executing_path.stem + f".{func.__name__}." + {
            "windows": "dll",
            "darwin": "dylib",
            "linux": "so",
        }.get(system_name, "so"))
        pycache_path.mkdir(exist_ok=True, parents=True)
        cpp_path = pycache_path / (executing_path.stem + f".{func.__name__}.cpp")
        cpp_path.write_text(cpp_code, encoding="utf-8")
        call: list[str] = [
            PATH_COMPILER,
            f"-std=c++{cxx}",
            "-shared",
            "-static",
            "-static-libgcc",
            "-static-libstdc++",
            f"-O{o}",
            str(cpp_path),
            "-o", str(output),
        ]
        result = subprocess.run(call, capture_output=True, text=True)
        cpp_path.unlink(missing_ok=True)
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{result.stderr}")

        type_wrap: dict[str, type] = {
            "int": ctypes.c_int,
            "double": ctypes.c_double,
            "float": ctypes.c_float,
            "long": ctypes.c_long,
            "char": ctypes.c_char,
            "long long": ctypes.c_longlong,
            "unsigned int": ctypes.c_uint,
            "unsigned long": ctypes.c_ulong,
            "unsigned long long": ctypes.c_ulonglong,
        }
        argtypes: list[type] = []
        for arg in type_fn.args:
            arg_type = parse_type(arg[1], cpp_data.type_ctx, cpp_data.state)
            argtypes.append(type_wrap[arg_type])
        
        jitFn = JitFunction(func, str(output), type_wrap[type_ret], argtypes)
        @wraps(func)
        def func_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return jitFn(*args, **kwargs)
        return func_wrapper
    
    if func is not None:
        return wrapper(func)
    
    return wrapper