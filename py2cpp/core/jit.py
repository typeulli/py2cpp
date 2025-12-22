import ast
import ctypes
import hashlib
import inspect
import importlib.util
import subprocess
import sys
import sysconfig

from ctypes import wintypes, CDLL
from dataclasses import dataclass
from functools import wraps
from importlib.metadata import version
from json import load
from pathlib import Path
from typing import Any, Callable, Generic, ParamSpec, TypeVar, overload

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
    "darwin": Path("/usr/bin/vswhere")
}

VCVARS = {
    "windows": "vcvars64.bat",
    "linux": "vcvars64.sh",
    "darwin": "vcvars64.sh"
}

EXTENSIONS = {
    "windows": ".dll",
    "linux": ".so",
    "darwin": ".dylib"
}

def get_extension(system_name: str) -> str:
    if system_name not in EXTENSIONS:
        raise ValueError(f"Unsupported system: {system_name}")
    return EXTENSIONS[system_name]

def remove_decorators(source: str) -> tuple[ast.FunctionDef, str]:
    astNode = ast.parse(source)
    if not astNode.body:
        raise ValueError("No function definition found in source.")
    funcNode = assert_type(astNode.body[0], ast.FunctionDef)
    while funcNode.decorator_list:
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
    return funcNode, source

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

CTYPE_MAP: dict[str, type] = {
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

@dataclass(frozen=True)
class JitHeader:
    source: str

    system: str
    cxx: int
    o: int

    @property
    def hash(self):
        source = ast.dump(ast.parse(self.source), annotate_fields=True, include_attributes=False)
        
        hasher = hashlib.md5()

        hasher.update(f"{version('py2cpp')}:{self.system}:{self.cxx}:{self.o}".encode("utf-8"))
        hasher.update(source.encode("utf-8"))
        return hasher
    
    
    @property
    def path_meta(self) -> Path:
        py2cpp_setting.PATH_CACHE_JIT.mkdir(exist_ok=True, parents=True)
        path_meta = py2cpp_setting.PATH_CACHE_JIT / f"{self.hash.hexdigest()}.json"
        return path_meta
    
    
    def __hash__(self) -> int:
        return int(self.hash.hexdigest(), 16)
@dataclass(frozen=True)
class JitInfo:
    
    @staticmethod
    def load(header: JitHeader) -> "JitInfo | None":
        path_meta = header.path_meta
        if not path_meta.exists():
            return None
        with path_meta.open("r", encoding="utf-8") as f:
            data: dict[str, Any] = load(f)
        
        assert type(data) == dict
        assert "type_return" in data and "type_arguments" in data and "py_mode" in data, "Invalid JIT header file."
        data_type_return: Any = data["type_return"]
        data_type_arguments: Any = data["type_arguments"]
        data_py_mode: Any = data["py_mode"]
        
        assert data_type_return in CTYPE_MAP, f"Unsupported return type: {data_type_return}"
        
        type_return = CTYPE_MAP[data_type_return]
        type_arguments: list[type] = [CTYPE_MAP[argt] for argt in data_type_arguments]
        py_mode = assert_type(data_py_mode, bool)

        return JitInfo(
            header=header,
            type_return=type_return,
            type_arguments=type_arguments,
            py_mode=py_mode,
        )
    
    header: JitHeader
    type_return: type
    type_arguments: list[type]
    py_mode: bool
    
    @property
    def path(self) -> Path:
        py2cpp_setting.PATH_CACHE_JIT.mkdir(exist_ok=True, parents=True)
        path_clib = py2cpp_setting.PATH_CACHE_JIT / f"{self.header.hash.hexdigest()}.pyd"
        if not self.py_mode:
            path_clib = path_clib.with_suffix(get_extension(self.header.system))
        return path_clib
    
    def save_meta(self) -> None:
        import json
        data: dict[str, Any] = {
            "type_return": [k for k, v in CTYPE_MAP.items() if v == self.type_return][0],
            "type_arguments": [[k for k, v in CTYPE_MAP.items() if v == argt][0] for argt in self.type_arguments],
            "py_mode": self.py_mode,
        }
        with self.header.path_meta.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    
    def __post_init__(self):
        self.save_meta()
    
    def __hash__(self) -> int:
        return hash(self.header)

class JitFunction(Generic[P, T]):
    
    @staticmethod
    def create(func: Callable[CP, CT], info: JitInfo) -> "JitFunction[CP, CT]":
        if info.py_mode:
            return JitPydFunction(func, info)
        else:
            return JitDllFunction(func, info)
    
    @property
    def wrapper(self) -> Callable[P, T]:
        raise NotImplementedError()
    
    def __init__(self, func: Callable[P, T], info: JitInfo) -> None:
        self.func = func
        self.info = info

class JitDllFunction(Generic[P, T], JitFunction[P, T]):
    def __init__(self, func: Callable[P, T], info: JitInfo) -> None:
        super().__init__(func, info)
        
        self.lib = CDLL(str(info.path))
        self.handle = self.lib._handle
        
        self.cfunc = getattr(self.lib, func.__name__)
        self.cfunc.restype = info.type_return
        self.cfunc.type_arguments = info.type_arguments
        wraps(func)(self)
        
    
    @property
    def wrapper(self) -> Callable[P, T]:
        @wraps(self.func)
        def func_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return self(*args, **kwargs)
        return func_wrapper

    def unload(self):
        match self.info.header.system:
            case "windows":
                if hasattr(self, "handle"):
                    ctypes.windll.kernel32.FreeLibrary(wintypes.HMODULE(self.handle))
                    del self.handle
            case _:
                if hasattr(self, "handle"):
                    ctypes.cdll.LoadLibrary(str(self.info.path)).dlclose(self.handle)
                    del self.handle

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.cfunc(*args, **kwargs)

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass

class JitPydFunction(Generic[P, T], JitFunction[P, T]):
    def __init__(self, func: Callable[P, T], info: JitInfo) -> None:
        super().__init__(func, info)
        
        spec = importlib.util.spec_from_file_location(f"py2cpp_lib_{info.header.hash.hexdigest()}", info.path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load JIT module from {info.path}")
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
    compile_command: list[str] = [
        py2cpp_setting.PATH_COMPILER,
        f"-std=c++{cxx}",
        *FLAGS.get(py2cpp_setting.SYSTEM_NAME, []),
        f"-O{o}",
        str(cpp_path),
        "-o", str(output),
    ]
    return subprocess.run(compile_command, capture_output=True, text=True)

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
        funcNode, source = remove_decorators(source)
        
        jitHeader = JitHeader(
            source=ast.unparse(funcNode),
            
            system=py2cpp_setting.SYSTEM_NAME,
            cxx=cxx,
            o=o
        )
        jitInfo_Test = JitInfo.load(jitHeader)
        if jitInfo_Test is not None:
            jitFn: JitFunction[CP, CT] = JitFunction[CP, CT].create(func, jitInfo_Test)
            return jitFn.wrapper
        
        cpp_data = py_2_cpp("from py2cpp.utils.header import *\n" + source, path=f"<jit:{func.__name__}>", setting=setting)
        cpp_code = cpp_data.code
        
        
        
        
        _type_fn = cpp_data.type_ctx.get_vartype(f"{func.__name__}")
        assert type(_type_fn) == FunctionTypeData
        type_fn: FunctionTypeData = _type_fn
        type_return = parse_type(type_fn.return_type, cpp_data.type_ctx, cpp_data.state)
        fn_code = type_return + " " + func.__name__ + "("
        for i, arg in enumerate(type_fn.args):
            arg_type = parse_type(arg[1], cpp_data.type_ctx, cpp_data.state)
            fn_code += arg_type + " " + arg[0]
            if i != len(type_fn.args) - 1:
                fn_code += ", "
        fn_code += ")"
        
        
        
        type_arguments: list[type] = []
        for arg in type_fn.args:
            arg_type = parse_type(arg[1], cpp_data.type_ctx, cpp_data.state)
            type_arguments.append(CTYPE_MAP[arg_type])
        
        
        _PY_MODE = type_return == "PyObject*"
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
            header=jitHeader,
            
            type_return=CTYPE_MAP[type_return],
            type_arguments=type_arguments,
            py_mode=_PY_MODE
        )
        
        
        if _PY_MODE:
            cpp_code += PY_MODULE_TEMPLATE.format(
                name_fn=func.__name__,
                name_struct=f"py2cpp_struct_{jitHeader.hash.hexdigest()}",
                name_lib=f"py2cpp_lib_{jitHeader.hash.hexdigest()}",
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
        
        jitFn = JitFunction[CP, CT].create(func, jitInfo)
        return jitFn.wrapper
    
    if func is not None:
        return wrapper(func)
    
    return wrapper