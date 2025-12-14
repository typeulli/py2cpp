import ast
import ctypes
import hashlib
import inspect
import subprocess

from ctypes import wintypes, CDLL
from dataclasses import dataclass
from functools import wraps
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Generic, ParamSpec, TypeVar, overload


from py2cpp import setting
from py2cpp.core.compiler import Setting, py_2_cpp, FunctionTypeData, parse_type
from py2cpp.utils.code import assert_type
from py2cpp.utils.parse import indexMultiLine


P = ParamSpec("P")
T = TypeVar("T")
CP = ParamSpec("CP")
CT = TypeVar("CT")

FLAGS = {
    "windows": ["-shared"],
    "linux": ["-shared"],
    "darwin": ["-dynamiclib"]
}

def get_extension(system_name: str) -> str:
    match system_name:
        case "windows":
            return "dll"
        case "darwin":
            return "dylib"
        case _:
            return "so"

def remove_decorators(source: str, node: ast.FunctionDef | None = None) -> str:
    if node is None:
        astNode = ast.parse(source)
        if not astNode.body:
            return source
        funcNode = assert_type(astNode.body[0], ast.FunctionDef)
    else:
        funcNode = node
    if not funcNode.decorator_list:
        return source
    while True:
        if not funcNode.decorator_list:
            break
        decoratorNode = funcNode.decorator_list.pop(0)
        end_lineno = decoratorNode.end_lineno or decoratorNode.lineno
        source_decorator = indexMultiLine(
            source,
            decoratorNode.lineno - 1,
            decoratorNode.col_offset,
            end_lineno - 1,
            decoratorNode.end_col_offset or len(source.splitlines()[end_lineno - 1])
        )
        source = source.replace(source_decorator + "\n", "", 1)
    return source

@dataclass
class JitInfo:
    source: str
    type_ret: type
    argtypes: list[type]

    system: str
    cxx: int
    o: int

    @property
    def hash(self):
        print(self.type_ret, self.argtypes, self.system, self.cxx, self.o, self.source)
        source = ast.dump(ast.parse(self.source), annotate_fields=True, include_attributes=False)
        
        hasher = hashlib.sha256()

        hasher.update(f"{version('py2cpp')}:{self.system}:{self.cxx}:{self.o}:{self.type_ret!r}".encode("utf-8"))
        for argtype in self.argtypes:
            hasher.update(f":{argtype!r}".encode("utf-8"))
        hasher.update(source.encode("utf-8"))
        return hasher
    
    @property
    def path(self) -> Path:
        setting.PATH_CACHE.mkdir(exist_ok=True, parents=True)
        path_clib = setting.PATH_CACHE / f"{hash(self):x}.{get_extension(self.system)}"
        return path_clib
    
    def __hash__(self) -> int:
        return int(self.hash.hexdigest(), 16)

class JitFunction(Generic[P, T]):
    
    @staticmethod
    def fromCache(func: Callable[CP, CT], info: JitInfo) -> "JitFunction[CP, CT] | None":
        path_clib = info.path
        if not path_clib.exists():
            return None
        return JitFunction(func, str(path_clib), info)
        
        

    def __init__(self, func: Callable[P, T], path_clib: str, info: JitInfo) -> None:
        self.lib = CDLL(path_clib)
        self.handle = self.lib._handle
        self.path = path_clib
        
        self.func: Callable[P, T] = func
        self.cfunc = getattr(self.lib, func.__name__)
        self.cfunc.restype = info.type_ret
        self.cfunc.argtypes = info.argtypes
        wraps(func)(self)
        
        self.info = info
    
    @property
    def wrapper(self) -> Callable[P, T]:
        @wraps(self.func)
        def func_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return self(*args, **kwargs)
        return func_wrapper

    def unload(self):
        match self.info.system:
            case "windows":
                if hasattr(self, "handle"):
                    ctypes.windll.kernel32.FreeLibrary(wintypes.HMODULE(self.handle))
                    del self.handle
            case _:
                if hasattr(self, "handle"):
                    ctypes.cdll.LoadLibrary(self.path).dlclose(self.handle)
                    del self.handle

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.cfunc(*args, **kwargs)

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass

@overload
def jit(func: Callable[P, T], *, cxx: int = 17, o: int = 3) -> Callable[P, T]: ...
@overload
def jit(func: None = None, *, cxx: int = 17, o: int = 3) -> Callable[[Callable[P, T]], Callable[P, T]]: ...
def jit(func: Callable[P, T] | None = None, *, cxx: int = 17, o: int = 3) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    def wrapper(func: Callable[P, T]) -> Callable[P, T]:
        DELSPEC = "__declspec(dllexport)" if setting.SYSTEM_NAME == "windows" else "__attribute__((visibility(\"default\")))"
        
        source = inspect.getsource(func)
        source = remove_decorators(source)
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
        cpp_code = cpp_code.replace(fn_code, f"extern \"C\" {DELSPEC} {fn_code}", 1)
        cpp_code = cpp_code[:cpp_code.rindex("int main()")].strip()
        
        
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



        
        jitInfo = JitInfo(
            source=source,
            type_ret=type_wrap[type_ret],
            argtypes=argtypes,
            system=setting.SYSTEM_NAME,
            cxx=cxx,
            o=o
        )
        jitCache = JitFunction.fromCache(func, jitInfo)
        if jitCache is not None:
            return jitCache.wrapper
        



        output = jitInfo.path
        cpp_path = output.with_suffix(".cpp")
        cpp_path.write_text(cpp_code, encoding="utf-8")
        call: list[str] = [
            setting.PATH_COMPILER,
            f"-std=c++{cxx}",
            *FLAGS.get(setting.SYSTEM_NAME, []),
            f"-O{o}",
            str(cpp_path),
            "-o", str(output),
        ]
        result = subprocess.run(call, capture_output=True, text=True)
        cpp_path.unlink(missing_ok=True)
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{result.stderr}")
        
        jitFn = JitFunction(func, str(output), jitInfo)
        return jitFn.wrapper
    
    if func is not None:
        return wrapper(func)
    
    return wrapper