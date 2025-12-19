import ast
import ctypes
import hashlib
import inspect
import subprocess
import sys
import sysconfig

from ctypes import wintypes, CDLL
from dataclasses import dataclass
from functools import wraps
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Generic, ParamSpec, TypeVar, overload


from py2cpp import setting as py2cpp_setting
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

VSWHERE_PATH = {
    "windows": Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"),
    "linux": Path("/usr/bin/vswhere"),
    "darwin": Path("/usr/bin/vswhere"),
}

VCVARS = {
    "windows": "vcvars64.bat",
    "linux": "vcvars64.sh",
    "darwin": "vcvars64.sh",
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
            decoratorNode.col_offset - 1,
            end_lineno - 1,
            decoratorNode.end_col_offset or len(source.splitlines()[end_lineno - 1]) - 1
        )
        source = source.replace(source_decorator + "\n", "", 1)
    return source

PY_MODULE_TEMPLATE = """

static PyMethodDef MyLibMethods[] = {{
    {{
        "{name_fn}",
        {name_fn},
        METH_VARARGS,
        {description}
    }},
    {{NULL, NULL, 0, NULL}}
}};

static struct PyModuleDef {name_struct} = {{
    PyModuleDef_HEAD_INIT,
    "{name_lib}",
    NULL,
    -1,
    MyLibMethods
}};

PyMODINIT_FUNC PyInit_{name_lib}(void) {{
    return PyModule_Create(&{name_struct});
}}
"""

@dataclass
class JitInfo:
    source: str
    type_ret: type
    argtypes: list[type]
    py_mode: bool

    system: str
    cxx: int
    o: int

    @property
    def hash(self):
        source = ast.dump(ast.parse(self.source), annotate_fields=True, include_attributes=False)
        
        hasher = hashlib.md5()

        hasher.update(f"{version('py2cpp')}:{self.system}:{self.cxx}:{self.o}:{self.type_ret!r}:{self.py_mode}".encode("utf-8"))
        for argtype in self.argtypes:
            hasher.update(f":{argtype!r}".encode("utf-8"))
        hasher.update(source.encode("utf-8"))
        return hasher
    
    @property
    def path(self) -> Path:
        py2cpp_setting.PATH_CACHE_JIT.mkdir(exist_ok=True, parents=True)
        path_clib = py2cpp_setting.PATH_CACHE_JIT / f"{self.hash.hexdigest()}.{get_extension(self.system) if not self.py_mode else 'pyd'}"
        return path_clib
    
    def __hash__(self) -> int:
        return int(self.hash.hexdigest(), 16)

class JitFunction(Generic[P, T]):
    
    @staticmethod
    def create(func: Callable[CP, CT], path_clib: str, info: JitInfo) -> "JitFunction[CP, CT]":
        if info.py_mode:
            return JitPydFunction(func, str(path_clib), info)
        else:
            return JitDllFunction(func, str(path_clib), info)
    
    @property
    def wrapper(self) -> Callable[P, T]:
        raise NotImplementedError()
    
    def __init__(self, func: Callable[P, T], path_clib: str, info: JitInfo) -> None:
        self.func = func
        self.path = path_clib
        self.info = info

class JitDllFunction(Generic[P, T], JitFunction[P, T]):
    def __init__(self, func: Callable[P, T], path_clib: str, info: JitInfo) -> None:
        super().__init__(func, path_clib, info)
        self.lib = CDLL(path_clib)
        self.handle = self.lib._handle
        
        self.cfunc = getattr(self.lib, func.__name__)
        self.cfunc.restype = info.type_ret
        self.cfunc.argtypes = info.argtypes
        wraps(func)(self)
        
    
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

class JitPydFunction(Generic[P, T], JitFunction[P, T]):
    def __init__(self, func: Callable[P, T], path_clib: str, info: JitInfo) -> None:
        super().__init__(func, path_clib, info)
        
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"py2cpp_lib_{info.hash.hexdigest()}", path_clib)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load JIT module from {path_clib}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.cfunc = getattr(module, func.__name__)
    
    @property
    def wrapper(self) -> Callable[P, T]:
        return self.cfunc
    
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.cfunc(*args, **kwargs)

def compile_dll(cxx: int, o: int, cpp_path: Path, output: Path) -> subprocess.CompletedProcess[str]:
    call: list[str] = [
        py2cpp_setting.PATH_COMPILER,
        f"-std=c++{cxx}",
        *FLAGS.get(py2cpp_setting.SYSTEM_NAME, []),
        f"-O{o}",
        str(cpp_path),
        "-o", str(output),
    ]
    return subprocess.run(call, capture_output=True, text=True)

def find_vcvars64() -> Path:
    vswhere = Path(VSWHERE_PATH[py2cpp_setting.SYSTEM_NAME])
    if not vswhere.exists():
        raise RuntimeError("vswhere.exe not found")

    result = subprocess.run(
        [str(vswhere), "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
        capture_output=True,
        text=True
    )

    path = result.stdout.strip()
    if not path:
        raise RuntimeError("No Visual Studio installation found")

    vcvars = Path(path) / "VC" / "Auxiliary" / "Build" / VCVARS[py2cpp_setting.SYSTEM_NAME]
    if not vcvars.exists():
        raise RuntimeError(f"{VCVARS[py2cpp_setting.SYSTEM_NAME]} not found in {vcvars}")

    return vcvars

def compile_pyd(cxx: int, o: int, cpp_path: Path, output: Path):
    include_py = sysconfig.get_config_var("INCLUDEPY")
    base = Path(sysconfig.get_config_var("installed_base"))
    lib_path = base / "libs"

    cmd_line = (
        f'call "{find_vcvars64()}" && '
        f'{py2cpp_setting.PATH_COMPILER_PYD} /std:c++{cxx} /LD '
        f'{" /O2" if o >= 2 else "/Od"} '
        f'/I"{include_py}" "{cpp_path}" '
        f'/link /LIBPATH:"{lib_path}" python311.lib /OUT:"{output}"'
    )

    result = subprocess.run(f'cmd /c "{cmd_line}"', shell=True, text=True, cwd=str(cpp_path.parent), capture_output=True)
    
    cpp_path.with_suffix(".obj").unlink(missing_ok=True)
    cpp_path.with_suffix(".lib").unlink(missing_ok=True)
    cpp_path.with_suffix(".exp").unlink(missing_ok=True)
    
    return result

@overload
def jit(func: Callable[CP, CT], *, cxx: int = 17, o: int = 3) -> Callable[CP, CT]: ...
@overload
def jit(func: None = None, *, cxx: int = 17, o: int = 3) -> Callable[[Callable[CP, CT]], Callable[CP, CT]]: ...
def jit(func: Callable[CP, CT] | None = None, *, cxx: int = 17, o: int = 3) -> Callable[CP, CT] | Callable[[Callable[CP, CT]], Callable[CP, CT]]:
    def wrapper(func: Callable[CP, CT]) -> Callable[CP, CT]:
        DELSPEC = "__declspec(dllexport)" if py2cpp_setting.SYSTEM_NAME == "windows" else "__attribute__((visibility(\"default\")))"
        setting = Setting()
        
        source = inspect.getsource(func)
        source = remove_decorators(source)
        cpp_data = py_2_cpp("from py2cpp.utils.header import *\n" + source, path=f"<jit:{func.__name__}>", setting=setting)
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
            "PyObject*": ctypes.py_object,
        }
        argtypes: list[type] = []
        for arg in type_fn.args:
            arg_type = parse_type(arg[1], cpp_data.type_ctx, cpp_data.state)
            argtypes.append(type_wrap[arg_type])
        
        
        _PY_MODE = type_ret == "PyObject*"
        if _PY_MODE:
            assert all(argt[1].is_py_object for argt in type_fn.args), "All types must either be PyObject*, or none of them must be PyObject*."
        else:
            assert all(not argt[1].is_py_object for argt in type_fn.args), "All types must either be PyObject*, or none of them must be PyObject*."


        assert fn_code in cpp_code
        if _PY_MODE:
            cpp_code = cpp_code.replace(
                    fn_code + " " + "{\n",
                    f"static PyObject* {func.__name__}(PyObject* self, PyObject* args) " + "{\n"
                    f"{setting.indent}PyObject " + ", ".join("*"+t[0] for t in type_fn.args) + ";\n"
                    f"{setting.indent}if (!PyArg_ParseTuple(args, \"{''.join('O' for _ in type_fn.args)}\", {', '.join('&'+t[0] for t in type_fn.args)})) {{\n"
                    f"{setting.indent*2}return nullptr;\n"
                    f"{setting.indent}}}\n",
                1
            )
        else:
            cpp_code = cpp_code.replace(
                fn_code,
                f"extern \"C\" {DELSPEC} {fn_code}" 
            , 1)

        
        cpp_code = cpp_code[:cpp_code.rindex("int main()")].strip()

        
        jitInfo = JitInfo(
            source=source,
            type_ret=type_wrap[type_ret],
            argtypes=argtypes,
            py_mode=_PY_MODE,
            
            system=py2cpp_setting.SYSTEM_NAME,
            cxx=cxx,
            o=o
        )
        
        path_clib = jitInfo.path
        if path_clib.exists():
            return JitFunction.create(func, str(path_clib), jitInfo).wrapper
        
        
        if _PY_MODE:
            cpp_code += PY_MODULE_TEMPLATE.format(
                name_fn=func.__name__,
                name_struct=f"py2cpp_struct_{jitInfo.hash.hexdigest()}",
                name_lib=f"py2cpp_lib_{jitInfo.hash.hexdigest()}",
                description=repr((func.__doc__ or "") + "'")[:-2] + '"'
            )



        output = jitInfo.path
        cpp_path = output.with_suffix(".cpp")
        cpp_path.write_text(cpp_code, encoding="utf-8")
        result = (compile_pyd if _PY_MODE else compile_dll)(cxx, o, cpp_path, output)
        cpp_path.unlink(missing_ok=True)
        if result.returncode != 0:
            error = "Compilation failed:"
            if result.stdout:
                error += f"\n\n[stdout]\n{result.stdout}"
            if result.stderr:
                error += f"\n\n[stderr]\n{result.stderr}"
            error += f"\n\n[C++ code]\n{cpp_code}"
            raise RuntimeError(error)
        
        jitFn: JitFunction[CP, CT] = JitFunction[CP, CT].create(func, str(output), jitInfo)
        return jitFn.wrapper
    
    if func is not None:
        return wrapper(func)
    
    return wrapper