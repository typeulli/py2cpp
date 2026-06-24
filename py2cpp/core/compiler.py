import ast
import random
import re
import shutil
import tempfile
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from py2cpp.setting import PATH_COMPILER
from py2cpp.utils.type import FunctionTypeData, parse_types, TypeContext, TypeData
from py2cpp.utils.code import assert_type, get_time
from py2cpp.utils.parse import indexMultiLine, pad, unwrap_paren, CommentInfo, parse_comments


path_here = Path(__file__).parent

path_builtins = path_here / "builtins"

def _builtins_getIdx(last_idx: list[int] = [0]) -> int:
    v = last_idx[0]
    last_idx[0] += 1
    return v

@dataclass
class Builtins:
    include: str
    inline: bool = False
    requires: list["Builtins"] = field(default_factory=lambda: [])

    idx: int = field(default_factory=_builtins_getIdx)

    def toString(self, include_set: set[str]) -> str:
        if self.include in include_set:
            return ""

        include_set.add(self.include)

        result = ""

        for req in self.requires:
            result = req.toString(include_set) + result

        if self.inline:
            result += self.include + "\n"
        else:
            result += (path_builtins / str(self.include)).read_text(encoding="utf-8") + "\n"

        return result

    def __hash__(self):
        return hash(self.idx)

CStdDef = Builtins("#include <cstddef>", inline=True)
IOStream = Builtins("#include <iostream>", inline=True)
String = Builtins("#include <string>", inline=True)
List = Builtins("#include <vector>", inline=True)
Str_Find = Builtins("str/find.cpp", requires=[String])
CCtype = Builtins("#include <cctype>", inline=True)
Algorithm = Builtins("#include <algorithm>", inline=True)
CStdLib = Builtins("#include <cstdlib>", inline=True)
Tuple = Builtins("#include <tuple>", inline=True)
CMath = Builtins("#include <cmath>", inline=True)
Complex = Builtins("#include <complex>", inline=True)
Execution = Builtins("#include <execution>", inline=True)
Python = Builtins("#include <Python.h>", inline=True)
Random = Builtins("#include <random>", inline=True)
RandomEngine = Builtins(
    "static std::mt19937 __py2c_rng(std::random_device{}());\n",
    inline=True,
    requires=[Random]
)
Numeric = Builtins("#include <numeric>", inline=True)
Chrono = Builtins("#include <chrono>", inline=True)
Thread = Builtins("#include <thread>", inline=True)
CTime = Builtins("#include <ctime>", inline=True)
FStream = Builtins("#include <fstream>", inline=True)
SStream = Builtins("#include <sstream>", inline=True)
UnorderedMap = Builtins("#include <unordered_map>", inline=True)
UnorderedSet = Builtins("#include <unordered_set>", inline=True)

NameDict: dict[str, tuple[Builtins, str]] = {
    "string": (String, "std::string"),
    "vector": (List, "std::vector"),
    "tuple": (Tuple, "std::tuple"),
    "cin": (IOStream, "std::cin"),
    "cout": (IOStream, "std::cout"),
    "endl": (IOStream, "std::endl"),
    "getline": (IOStream, "std::getline"),
    "get": (Tuple, "std::get"),
    "transform": (Algorithm, "std::transform"),
    "toupper": (CCtype, "std::toupper"),
    "tolower": (CCtype, "std::tolower"),
    "flush": (IOStream, "std::flush"),
    "pow": (CMath, "std::pow"),
    "complex": (Complex, "std::complex"),
    "@python": (Python, ""),
}

MATH_FUNCS: dict[str, str] = {
    "sqrt": "std::sqrt", "sin": "std::sin", "cos": "std::cos", "tan": "std::tan",
    "asin": "std::asin", "acos": "std::acos", "atan": "std::atan", "atan2": "std::atan2",
    "sinh": "std::sinh", "cosh": "std::cosh", "tanh": "std::tanh",
    "log": "std::log", "log2": "std::log2", "log10": "std::log10",
    "exp": "std::exp", "pow": "std::pow",
    "floor": "std::floor", "ceil": "std::ceil", "trunc": "std::trunc",
    "fabs": "std::fabs", "fmod": "std::fmod",
    "gcd": "std::gcd",
}
MATH_CONSTS: dict[str, str] = {
    "pi": "3.14159265358979323846",
    "e": "2.71828182845904523536",
    "inf": "std::numeric_limits<double>::infinity()",
    "nan": "std::numeric_limits<double>::quiet_NaN()",
    "tau": "6.28318530717958647692",
}
Limits = Builtins("#include <limits>", inline=True)

TIME_FUNCS: set[str] = {"time", "sleep", "perf_counter", "monotonic"}

@dataclass
class DefaultTypesSetting:
    int_type: str = "long"
    float_type: str = "double"
    bool_type: str = "bool"

@dataclass
class Setting:
    indent: str = "    "
    use_py: bool = True
    py_to_comment: bool = False
    minimize_namespace: list[str] = field(default_factory=lambda: [])
    default_types: DefaultTypesSetting = field(default_factory=DefaultTypesSetting)

@dataclass
class Code:
    code: str
    comments: list[CommentInfo] = field(init=False)

    def __post_init__(self):
        self.comments = parse_comments(self.code)

    def popComments(self, expr: ast.stmt) -> tuple[list[CommentInfo], CommentInfo | None]:
        result: list[CommentInfo] = []
        remaining_comments: list[CommentInfo] = []
        for comment in self.comments:
            if comment.end_lineno < expr.lineno:
                result.append(comment)
            else:
                remaining_comments.append(comment)
        self.comments = remaining_comments
        remaining_comments = []
        
        sameline_comments: CommentInfo | None = None
        for comment in self.comments:
            if comment.lineno == expr.lineno:
                sameline_comments = comment
            else:
                remaining_comments.append(comment)
        self.comments = remaining_comments
        
        return result, sameline_comments

@dataclass
class State:
    origin_code: Code
    global_code: str = ""
    defined: dict[str, bool] = field(default_factory=lambda: dict[str, bool]())
    used_builtins: set[Builtins] = field(default_factory=lambda: set[Builtins]())
    used_tempids: set[str] = field(default_factory=lambda: set[str]())
    setting: Setting = field(default_factory=Setting)
    imports_from_random: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    imports_random_module: set[str] = field(default_factory=lambda: set[str]())
    imports_from_math: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    imports_math_module: set[str] = field(default_factory=lambda: set[str]())
    imports_from_time: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    imports_time_module: set[str] = field(default_factory=lambda: set[str]())
    file_vars: dict[str, str] = field(default_factory=lambda: dict[str, str]())

    def get_tempid(self) -> str:
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
        while True:
            id_ = "".join(random.choice(chars) for _ in range(8))
            if id_[0] not in "0123456789" and id_ not in self.used_tempids:
                self.used_tempids.add(id_)
                return id_

    def get_name(self, base: str) -> str:
        builtin, name = NameDict[base]
        self.used_builtins.add(builtin)
        if base in self.setting.minimize_namespace:
            return base
        return name

@dataclass
class StmtState:
    state: State
    extra_codes_pre: list[str] = field(default_factory=lambda: [])
    extra_codes_post: list[str] = field(default_factory=lambda: [])

    @property
    def collect_pre_extra(self) -> str:
        return "\n".join(self.extra_codes_pre)

    @property
    def collect_post_extra(self) -> str:
        return "\n".join(self.extra_codes_post)

@dataclass
class ScriptTemplateAssign:
    type: str
    name: str
    expr: str


class ScriptTemplate(metaclass=ABCMeta):
    @abstractmethod
    def format(self, stmt_state: StmtState, **kwargs: str) -> str: ...

class ScriptTemplateSimple(ScriptTemplate):
    def __init__(self, pre: str, result_expr: str, post: str="", require_vars: list[str] | None = None, require_names: list[str] | None = None) -> None:
        self.pre = pre.lstrip("\n").rstrip("\n")
        self.post = post.lstrip("\n").rstrip("\n")
        self.result_expr = result_expr
        self.require_vars = require_vars if require_vars is not None else []
        self.require_tmps: list[str] = re.findall(r"\{(_[a-zA-Z][a-zA-Z0-9_]*)\}", self.pre) + re.findall(r"\{(_[a-zA-Z][a-zA-Z0-9_]*)\}", self.result_expr)
        self.require_names: list[str] = (require_names or []) + re.findall(r"\{__([a-zA-Z][a-zA-Z0-9_]*)\}", self.pre) + re.findall(r"\{__([a-zA-Z][a-zA-Z0-9_]*)\}", self.result_expr)

    def format(self, stmt_state: StmtState, **kwargs: str) -> str:
        state = stmt_state.state
        __get_tempid = lambda: state.get_tempid()

        tmp_vars = {tmp: __get_tempid() for tmp in self.require_tmps}
        for var in self.require_vars:
            assert var in kwargs, f"Missing variable '{var}' for template"

        names = {"__"+name: state.get_name(name) for name in self.require_names}

        stmt_state.extra_codes_pre.append(self.pre.format(**tmp_vars, **kwargs, **names))
        stmt_state.extra_codes_post.append(self.post.format(**tmp_vars, **kwargs, **names))
        return self.result_expr.format(**tmp_vars, **kwargs, **names)

Template_String_Upper = ScriptTemplateSimple(
    pre="""
{__string} {_tmp1} = {string};
{__transform}({_tmp1}.begin(), {_tmp1}.end(), {_tmp1}.begin(), {__toupper});
""",
    result_expr="{_tmp1}"
)

Template_String_Lower = ScriptTemplateSimple(
    pre="""
{__string} {_tmp1} = {string};
{__transform}({_tmp1}.begin(), {_tmp1}.end(), {_tmp1}.begin(), {__tolower});
""",
    result_expr="{_tmp1}"
)

Template_Pop_NoIndex = ScriptTemplateSimple(
    pre="""
auto {_tmp1} = {list};
auto {_tmp2} = {_tmp1}.back();
{_tmp1}.pop_back();
""",
    result_expr="{_tmp2}",
    require_vars=["list"]
)

Template_Pop_WithIndex = ScriptTemplateSimple(
    pre="""
auto {_tmp1} = {list};
auto {_tmp2} = {_tmp1}[{index}];
{_tmp1}.erase({_tmp1}.begin() + {index});
""",
    result_expr="{_tmp2}",
    require_vars=["list", "index"]
)

Template_Python_Call = ScriptTemplateSimple(
    pre="""
PyObject* {_tmp1} = PyObject_CallObject({func}, {args});
if ({_tmp1} == nullptr) {{ return nullptr; }}
""",
    result_expr="{_tmp1}",
    post = """
Py_DECREF({_tmp1});
""",
    require_vars=["func", "args"],
    require_names=["@python"]
)

Template_Python_Getattr = ScriptTemplateSimple(
    pre="""
PyObject* {_tmp1} = PyObject_GetAttrString({obj}, {attr_name});
if ({_tmp1} == nullptr) {{ return nullptr; }}
""",
    result_expr="{_tmp1}",
    post = """
Py_DECREF({_tmp1});
""",
    require_vars=["obj", "attr_name"],
    require_names=["@python"]
)

Template_Python_Setattr = ScriptTemplateSimple(
    pre="""
int {_tmp1} = PyObject_SetAttrString({obj}, {attr_name}, {value});
if ({_tmp1} != 0) {{ return nullptr; }}
""",
    result_expr="",
    require_vars=["obj", "attr_name", "value"],
    require_names=["@python"]
)

Template_Randint = ScriptTemplateSimple(
    pre="",
    result_expr="std::uniform_int_distribution<long>({a}, {b})(__py2c_rng)",
    require_vars=["a", "b"]
)

Template_Randrange1 = ScriptTemplateSimple(
    pre="",
    result_expr="std::uniform_int_distribution<long>(0, ({stop}) - 1)(__py2c_rng)",
    require_vars=["stop"]
)

Template_Randrange2 = ScriptTemplateSimple(
    pre="",
    result_expr="std::uniform_int_distribution<long>({start}, ({stop}) - 1)(__py2c_rng)",
    require_vars=["start", "stop"]
)

Template_Randrange3 = ScriptTemplateSimple(
    pre="""
long {_start} = {start};
long {_stop} = {stop};
long {_step} = {step};
""",
    result_expr="({_start} + std::uniform_int_distribution<long>(0, ({_stop} - {_start} - ({_step} > 0 ? 1 : -1)) / {_step})(__py2c_rng) * {_step})",
    require_vars=["start", "stop", "step"]
)

Template_Shuffle = ScriptTemplateSimple(
    pre="",
    result_expr="std::shuffle({v}.begin(), {v}.end(), __py2c_rng)",
    require_vars=["v"]
)

Template_Sorted = ScriptTemplateSimple(
    pre="""
auto {_tmp1} = {iterable};
std::sort({_tmp1}.begin(), {_tmp1}.end());
""",
    result_expr="{_tmp1}",
    require_vars=["iterable"]
)

Template_Sorted_Reverse = ScriptTemplateSimple(
    pre="""
auto {_tmp1} = {iterable};
std::sort({_tmp1}.begin(), {_tmp1}.end(), std::greater<>());
""",
    result_expr="{_tmp1}",
    require_vars=["iterable"]
)

Template_Reversed = ScriptTemplateSimple(
    pre="""
auto {_tmp1} = {iterable};
std::reverse({_tmp1}.begin(), {_tmp1}.end());
""",
    result_expr="{_tmp1}",
    require_vars=["iterable"]
)

Template_Round_NDigits = ScriptTemplateSimple(
    pre="""
double {_factor} = std::pow(10.0, {ndigits});
""",
    result_expr="(std::round(({x}) * {_factor}) / {_factor})",
    require_vars=["x", "ndigits"]
)

Template_TimeSleep = ScriptTemplateSimple(
    pre="",
    result_expr="std::this_thread::sleep_for(std::chrono::duration<double>({seconds}))",
    require_vars=["seconds"]
)

Template_PerfCounter = ScriptTemplateSimple(
    pre="",
    result_expr="std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()"
)

Template_Time = ScriptTemplateSimple(
    pre="",
    result_expr="static_cast<double>(std::time(nullptr))"
)

Template_FileRead = ScriptTemplateSimple(
    pre="""
std::stringstream {_tmp1};
{_tmp1} << {file}.rdbuf();
""",
    result_expr="{_tmp1}.str()",
    require_vars=["file"]
)

Template_FileReadline = ScriptTemplateSimple(
    pre="""
std::string {_tmp1};
std::getline({file}, {_tmp1});
""",
    result_expr="{_tmp1}",
    require_vars=["file"]
)

Template_StrSplit = ScriptTemplateSimple(
    pre="""
std::vector<std::string> {_tmp1};
{{
    const std::string& {_s} = {string};
    const std::string& {_d} = {sep};
    size_t {_a} = 0, {_p};
    while (({_p} = {_s}.find({_d}, {_a})) != std::string::npos) {{
        {_tmp1}.push_back({_s}.substr({_a}, {_p} - {_a}));
        {_a} = {_p} + {_d}.size();
    }}
    {_tmp1}.push_back({_s}.substr({_a}));
}}
""",
    result_expr="{_tmp1}",
    require_vars=["string", "sep"]
)

Template_StrSplitWs = ScriptTemplateSimple(
    pre="""
std::vector<std::string> {_tmp1};
{{
    std::stringstream {_ss}({string});
    std::string {_it};
    while ({_ss} >> {_it}) {_tmp1}.push_back({_it});
}}
""",
    result_expr="{_tmp1}",
    require_vars=["string"]
)

Template_StrJoin = ScriptTemplateSimple(
    pre="""
std::string {_tmp1};
{{
    const auto& {_v} = {iterable};
    const std::string& {_sep} = {sep};
    for (size_t {_i} = 0; {_i} < {_v}.size(); ++{_i}) {{
        if ({_i} > 0) {_tmp1} += {_sep};
        {_tmp1} += {_v}[{_i}];
    }}
}}
""",
    result_expr="{_tmp1}",
    require_vars=["sep", "iterable"]
)

Template_StrReplace = ScriptTemplateSimple(
    pre="""
std::string {_tmp1} = {string};
{{
    const std::string& {_o} = {old};
    const std::string& {_n} = {new_};
    size_t {_p} = 0;
    while (({_p} = {_tmp1}.find({_o}, {_p})) != std::string::npos) {{
        {_tmp1}.replace({_p}, {_o}.size(), {_n});
        {_p} += {_n}.size();
    }}
}}
""",
    result_expr="{_tmp1}",
    require_vars=["string", "old", "new_"]
)

Template_StrStrip = ScriptTemplateSimple(
    pre="""
std::string {_tmp1};
{{
    const std::string& {_s} = {string};
    auto {_a} = {_s}.find_first_not_of(" \\t\\n\\r\\f\\v");
    auto {_b} = {_s}.find_last_not_of(" \\t\\n\\r\\f\\v");
    {_tmp1} = ({_a} == std::string::npos) ? std::string() : {_s}.substr({_a}, {_b} - {_a} + 1);
}}
""",
    result_expr="{_tmp1}",
    require_vars=["string"]
)

Template_StrLStrip = ScriptTemplateSimple(
    pre="""
std::string {_tmp1};
{{
    const std::string& {_s} = {string};
    auto {_a} = {_s}.find_first_not_of(" \\t\\n\\r\\f\\v");
    {_tmp1} = ({_a} == std::string::npos) ? std::string() : {_s}.substr({_a});
}}
""",
    result_expr="{_tmp1}",
    require_vars=["string"]
)

Template_StrRStrip = ScriptTemplateSimple(
    pre="""
std::string {_tmp1};
{{
    const std::string& {_s} = {string};
    auto {_b} = {_s}.find_last_not_of(" \\t\\n\\r\\f\\v");
    {_tmp1} = ({_b} == std::string::npos) ? std::string() : {_s}.substr(0, {_b} + 1);
}}
""",
    result_expr="{_tmp1}",
    require_vars=["string"]
)

Template_StrCount = ScriptTemplateSimple(
    pre="""
long {_tmp1} = 0;
{{
    const std::string& {_s} = {string};
    const std::string& {_sub} = {sub};
    if (!{_sub}.empty()) {{
        size_t {_p} = 0;
        while (({_p} = {_s}.find({_sub}, {_p})) != std::string::npos) {{
            ++{_tmp1};
            {_p} += {_sub}.size();
        }}
    }}
}}
""",
    result_expr="{_tmp1}",
    require_vars=["string", "sub"]
)

Template_Slice_List = ScriptTemplateSimple(
    pre="""
auto {_tmp1} = {iterable};
""",
    result_expr="std::vector<typename std::decay<decltype({_tmp1})>::type::value_type>({_tmp1}.begin() + ({lo}), {_tmp1}.begin() + ({hi}))",
    require_vars=["iterable", "lo", "hi"]
)

class ScriptTemplateClsInput(ScriptTemplate):
    def __init__(self) -> None:
        self.Template_Input_GetValue = ScriptTemplateSimple(
            pre="""
{__string} {_tmp1};
{__getline}({__cin}, {_tmp1});
""",
            result_expr="{_tmp1}"
        )
        self.Template_Input_SaveValue = ScriptTemplateSimple(
            pre="{__getline}({__cin}, {var});",
            result_expr=""
        )
    
    def format(self, stmt_state: StmtState, **kwargs: str) -> str:
        if "prompt" in kwargs:
            stmt_state.extra_codes_pre.append(stmt_state.state.get_name("cout") + " << " + kwargs["prompt"] + " << " + stmt_state.state.get_name("flush") + ";")
            kwargs.pop("prompt")
        if "var" in kwargs:
            var = kwargs["var"]
            if var not in stmt_state.state.defined:
                stmt_state.state.defined[var] = True
                stmt_state.extra_codes_pre.append(f"{stmt_state.state.get_name('string')} {var};")
            self.Template_Input_SaveValue.format(stmt_state, **kwargs)
            return ""
        else:
            return self.Template_Input_GetValue.format(stmt_state, **kwargs)
        
ScriptTemplateInput = ScriptTemplateClsInput()


def eval_type(expr: ast.expr, type_ctx: TypeContext, state: State, path: str = "") -> TypeData:
    match type(expr):
        case ast.Name:
            assert type(expr) == ast.Name
            if expr.id in type_ctx.struct_dict:
                return TypeData(type_=expr.id)
            test_func = type_ctx.type_dict.get(expr.id)
            if type(test_func) == FunctionTypeData:
                return test_func
            type_ = type_ctx.type_dict[path + expr.id]
            assert type_ != "Any"
            return type_
        case ast.Call:
            assert type(expr) == ast.Call
            if type(expr.func) == ast.Attribute and type(expr.func.value) == ast.Name:
                base = expr.func.value.id
                attr = expr.func.attr
                if base in state.imports_math_module and attr in MATH_FUNCS:
                    return TypeData(type_="builtins.float")
                if base in state.imports_time_module and attr in TIME_FUNCS:
                    return TypeData(type_="builtins.float")
                if base in state.imports_random_module and attr in ("randint", "randrange"):
                    return TypeData(type_="builtins.int")
            assert type(expr.func) == ast.Name
            match expr.func.id:
                case "int": return TypeData(type_="builtins.int")
                case "float": return TypeData(type_="builtins.float")
                case "str": return TypeData(type_="builtins.str")
                case "bool": return TypeData(type_="builtins.bool")
                case "complex": return TypeData(type_="builtins.complex")
                case "c_short" | "c_ushort" | "c_int" | "c_uint" | "c_long" | "c_ulong" | "c_longlong" | "c_ulonglong" | "c_float" | "c_double" | "c_char" | "c_bool" | "c_void":
                    return TypeData(type_="py2cpp." + expr.func.id)
                case "py_object":
                    return TypeData(type_="py2cpp.py_object")
                case _:
                    if expr.func.id in state.imports_from_math and state.imports_from_math[expr.func.id] in MATH_FUNCS:
                        return TypeData(type_="builtins.float")
                    if expr.func.id in state.imports_from_time and state.imports_from_time[expr.func.id] in TIME_FUNCS:
                        return TypeData(type_="builtins.float")
                    if expr.func.id in state.imports_from_random and state.imports_from_random[expr.func.id] in ("randint", "randrange"):
                        return TypeData(type_="builtins.int")
                    func = eval_type(expr.func, type_ctx, state, path)
                    if func.is_py_object:
                        return TypeData(type_="py2cpp.py_object")
                    assert type(func) == FunctionTypeData
                    return func.return_type
        case ast.Constant:
            assert type(expr) == ast.Constant
            if type(expr.value) == int:
                return TypeData(type_="builtins.int")
            if type(expr.value) == float:
                return TypeData(type_="builtins.float")
            if type(expr.value) == str:
                return TypeData(type_="builtins.str")
            if type(expr.value) == bool:
                return TypeData(type_="builtins.bool")
            if type(expr.value) == complex:
                return TypeData(type_="builtins.complex")
        case ast.Compare | ast.BoolOp:
            return TypeData(type_="builtins.bool")
        case ast.BinOp:
            assert type(expr) == ast.BinOp
            left_type = eval_type(expr.left, type_ctx, state, path)
            right_type = eval_type(expr.right, type_ctx, state, path)
            if left_type.type_ == right_type.type_:
                return left_type
            if left_type.type_ == "builtins.float" or right_type.type_ == "builtins.float":
                return TypeData(type_="builtins.float")
            for _t in ("c_ulonglong", "c_longlong", "c_ulong", "c_long", "c_uint", "c_int", "c_ushort", "c_short"):
                if left_type.is_py2cpp_type(_t) or right_type.is_py2cpp_type(_t):
                    return TypeData(type_="py2cpp." + _t)
            return TypeData(type_="builtins.int")
        case ast.Attribute:
            assert type(expr) == ast.Attribute
            if (type(expr.value) == ast.Name
                and expr.value.id in state.imports_math_module
                and expr.attr in MATH_CONSTS):
                return TypeData(type_="builtins.float")
            value_type = eval_type(expr.value, type_ctx, state, path)
            if value_type.is_py_object:
                return TypeData(type_="py2cpp.py_object")
            if value_type.type_.split(".")[-1] in type_ctx.struct_dict:
                struct_def = type_ctx.struct_dict[value_type.type_.split(".")[-1]]
                if expr.attr in struct_def.fields:
                    return struct_def.fields[expr.attr].type_
                raise TypeError(f"Attribute '{expr.attr}' not found in struct '{value_type.type_}'")
            raise TypeError(f"Unsupported at eval_type: attr for '{value_type.type_}'")
        case ast.Subscript:
            assert type(expr) == ast.Subscript
            value_type = eval_type(expr.value, type_ctx, state, path)
            if type(expr.slice) == ast.Slice:
                return value_type
            if value_type.type_ == "builtins.str":
                return TypeData(type_="builtins.str")
            if value_type.type_.startswith("builtins.list") and len(value_type.generics) >= 1:
                gen = value_type.generics[0]
                if type(gen) == TypeData:
                    return gen
            if value_type.type_.startswith("builtins.dict") and len(value_type.generics) >= 2:
                gen = value_type.generics[1]
                if type(gen) == TypeData:
                    return gen
            return value_type
        case ast.UnaryOp:
            assert type(expr) == ast.UnaryOp
            return eval_type(expr.operand, type_ctx, state, path)
        case _:
            raise NotImplementedError(f"Unsupported AST node type for eval_type: {type(expr)} {ast.dump(expr)}")
    raise NotImplementedError(f"Unsupported AST node type for eval_type: {type(expr)} {ast.dump(expr)}")
type_alias = {
    "py2cpp.api.c_int": "int",
    "py2cpp.api.c_uint": "unsigned int",
    "py2cpp.api.c_short": "short",
    "py2cpp.api.c_ushort": "unsigned short",
    "py2cpp.api.c_long": "long",
    "py2cpp.api.c_ulong": "unsigned long",
    "py2cpp.api.c_longlong": "long long",
    "py2cpp.api.c_ulonglong": "unsigned long long",
    "py2cpp.api.c_float": "float",
    "py2cpp.api.c_double": "double",
    "py2cpp.api.c_char": "char",
    "py2cpp.api.c_bool": "bool",
    "py2cpp.api.c_void": "void",
    "builtins.int": "long",
    "builtins.float": "double",
    "builtins.bool": "bool",
    "builtins.complex": "std::complex<double>",
}

for _key, _value in type_alias.copy().items():
    type_alias[_key[_key.rfind(".")+1:]] = _value

def parse_type(data: TypeData, type_ctx: TypeContext, state: State) -> str:
    def __parse_type(data: TypeData) -> str:
        return parse_type(data, type_ctx, state)

    if data.type_ == "builtins.str":
        return state.get_name("string")

    if data.type_ in type_alias:
        if data.type_ == "builtins.complex":
            return state.get_name("complex") + "<double>"
        return type_alias[data.type_]

    if data.type_.startswith("builtins.list"):
        vector_type = data.generics[0]
        assert type(vector_type) == TypeData
        return state.get_name("vector") + "<" + __parse_type(vector_type) + ">"

    if data.type_.startswith("builtins.dict"):
        assert len(data.generics) == 2
        k_type = data.generics[0]
        v_type = data.generics[1]
        assert type(k_type) == TypeData and type(v_type) == TypeData
        state.used_builtins.add(UnorderedMap)
        return "std::unordered_map<" + __parse_type(k_type) + ", " + __parse_type(v_type) + ">"

    if data.type_.startswith("builtins.set"):
        assert len(data.generics) == 1
        v_type = data.generics[0]
        assert type(v_type) == TypeData
        state.used_builtins.add(UnorderedSet)
        return "std::unordered_set<" + __parse_type(v_type) + ">"

    if data.type_.startswith("tuple"):
        return state.get_name("tuple") + "<" + ", ".join(__parse_type(assert_type(t, TypeData)) for t in data.generics) + ">"

    if data.type_ in type_ctx.struct_dict:
        return type_ctx.struct_dict[data.type_].name

    dot_index = data.type_.find(".")
    if dot_index != -1:
        if dot_index > data.type_.find("["):
            return __parse_type(TypeData(type_=data.type_[dot_index+1:], generics=data.generics))
    
    if state.setting.use_py:
        return "PyObject*"

    raise ValueError(f"Unknown type: {data.type_}")

@dataclass
class DirectData:
    type_: Literal["none", "assign"] = "none"
    value: str = ""

def dfs_stmt(node: ast.expr, type_ctx: TypeContext, stmt_state: StmtState, path: str = "", *, direct: DirectData = DirectData()) -> str:

    state = stmt_state.state

    def __dfs_stmt(node: ast.expr) -> str:
        return dfs_stmt(node, type_ctx, stmt_state, path)

    def __emit_random_call(func_name: str, args: list[ast.expr]) -> str:
        state.used_builtins.add(RandomEngine)
        match func_name:
            case "randint":
                if len(args) != 2: raise SyntaxError("random.randint requires 2 arguments")
                return Template_Randint.format(stmt_state,
                    a=unwrap_paren(__dfs_stmt(args[0])),
                    b=unwrap_paren(__dfs_stmt(args[1])))
            case "randrange":
                if len(args) == 1:
                    return Template_Randrange1.format(stmt_state,
                        stop=unwrap_paren(__dfs_stmt(args[0])))
                if len(args) == 2:
                    return Template_Randrange2.format(stmt_state,
                        start=unwrap_paren(__dfs_stmt(args[0])),
                        stop=unwrap_paren(__dfs_stmt(args[1])))
                if len(args) == 3:
                    return Template_Randrange3.format(stmt_state,
                        start=unwrap_paren(__dfs_stmt(args[0])),
                        stop=unwrap_paren(__dfs_stmt(args[1])),
                        step=unwrap_paren(__dfs_stmt(args[2])))
                raise SyntaxError("random.randrange requires 1-3 arguments")
            case "shuffle":
                if len(args) != 1: raise SyntaxError("random.shuffle requires 1 argument")
                state.used_builtins.add(Algorithm)
                return Template_Shuffle.format(stmt_state, v=unwrap_paren(__dfs_stmt(args[0])))
        raise NotImplementedError(f"random.{func_name} is not supported")

    def __emit_math_call(func_name: str, args: list[ast.expr]) -> str:
        if func_name not in MATH_FUNCS:
            raise NotImplementedError(f"math.{func_name} is not supported")
        state.used_builtins.add(CMath)
        cpp_name = MATH_FUNCS[func_name]
        args_str = ", ".join(unwrap_paren(__dfs_stmt(a)) for a in args)
        return f"{cpp_name}({args_str})"

    def __emit_math_const(const_name: str) -> str:
        if const_name not in MATH_CONSTS:
            raise NotImplementedError(f"math.{const_name} is not supported")
        if const_name in ("inf", "nan"):
            state.used_builtins.add(Limits)
        return MATH_CONSTS[const_name]

    def __emit_time_call(func_name: str, args: list[ast.expr]) -> str:
        match func_name:
            case "time":
                state.used_builtins.add(CTime)
                return Template_Time.format(stmt_state)
            case "sleep":
                if len(args) != 1: raise SyntaxError("time.sleep requires 1 argument")
                state.used_builtins.add(Chrono)
                state.used_builtins.add(Thread)
                return Template_TimeSleep.format(stmt_state,
                    seconds=unwrap_paren(__dfs_stmt(args[0])))
            case "perf_counter" | "monotonic":
                state.used_builtins.add(Chrono)
                return Template_PerfCounter.format(stmt_state)
        raise NotImplementedError(f"time.{func_name} is not supported")

    def __emit_file_open(node: ast.Call) -> str:
        if len(node.args) < 1: raise SyntaxError("open requires at least 1 argument")
        path_arg = unwrap_paren(__dfs_stmt(node.args[0]))
        mode = "r"
        if len(node.args) >= 2:
            mode_node = node.args[1]
            if type(mode_node) == ast.Constant and type(mode_node.value) == str:
                mode = mode_node.value
        for kwd in node.keywords:
            if kwd.arg == "mode" and type(kwd.value) == ast.Constant and type(kwd.value.value) == str:
                mode = kwd.value.value
        state.used_builtins.add(FStream)
        cls = "std::ofstream" if ("w" in mode or "a" in mode) else "std::ifstream"
        flags = ""
        if "a" in mode:
            flags = ", std::ios::app"
        if "b" in mode:
            flags += ", std::ios::binary"
        if direct.type_ == "assign":
            var_name = direct.value
            state.file_vars[var_name] = var_name
            state.defined[path + var_name] = True
            stmt_state.extra_codes_pre.append(f"{cls} {var_name}({path_arg}{flags});")
            return ""
        raise SyntaxError("open() must be used in an assignment or 'with' statement")
    
    match type(node):
        case ast.Constant:
            assert type(node) == ast.Constant
            if type(node.value) == bool:
                return "true" if node.value else "false"
            if type(node.value) == str:
                return "\"" + node.value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t").replace("\b", "\\b").replace("\f", "\\f") + "\""
            if type(node.value) == complex:
                return state.get_name("complex") + "<double>(" + str(node.value.real) + ", " + str(node.value.imag) + ")"
            return state.origin_code.code.splitlines()[node.lineno - 1][node.col_offset:node.end_col_offset]

        case ast.Attribute:
            assert type(node) == ast.Attribute
            if (type(node.value) == ast.Name
                and node.value.id in state.imports_math_module
                and node.attr in MATH_CONSTS):
                return __emit_math_const(node.attr)
            type_data = eval_type(node.value, type_ctx, state, path)
            if direct.type_ != "assign" and type_data.is_py_object:
                get_attr = Template_Python_Getattr.format(
                    stmt_state,
                    obj=__dfs_stmt(node.value),
                    attr_name="\"" + node.attr + "\""
                )
                return get_attr
            if type_data.type_ == "builtins.complex":
                if node.attr in ("real", "imag"):
                    return __dfs_stmt(node.value) + "." + node.attr + "()"
            return __dfs_stmt(node.value) + "." + node.attr

        case ast.List:
            assert type(node) == ast.List
            return "{" + ", ".join(unwrap_paren(__dfs_stmt(elt)) for elt in node.elts) + "}"

        case ast.Dict:
            assert type(node) == ast.Dict
            pairs: list[str] = []
            for k, v in zip(node.keys, node.values):
                assert k is not None
                pairs.append("{" + unwrap_paren(__dfs_stmt(k)) + ", " + unwrap_paren(__dfs_stmt(v)) + "}")
            return "{" + ", ".join(pairs) + "}"

        case ast.Set:
            assert type(node) == ast.Set
            return "{" + ", ".join(unwrap_paren(__dfs_stmt(elt)) for elt in node.elts) + "}"

        case ast.Tuple:
            assert type(node) == ast.Tuple
            return "std::make_tuple(" + ", ".join(unwrap_paren(__dfs_stmt(elt)) for elt in node.elts) + ")"

        case ast.Call:
            assert type(node) == ast.Call
            func = node.func
            if type(func) == ast.Name:
                match func.id:
                    case "c_cast":
                        if len(node.args) != 2: raise SyntaxError("c_cast requires exactly two arguments")
                        type_str = node.args[0]
                        val = node.args[1]
                        assert type(type_str) == ast.Constant and type(type_str.value) == str
                        parsed_type = parse_type(TypeData.from_str(type_str.value), type_ctx, state)
                        return f"({parsed_type})({unwrap_paren(__dfs_stmt(val))})"
                    case "c_static_cast" | "c_reinterpret_cast":
                        assert len(node.args) == 2
                        type_str = node.args[0]
                        val = node.args[1]
                        assert type(type_str) == ast.Constant and type(type_str.value) == str
                        parsed_type = parse_type(TypeData.from_str(type_str.value), type_ctx, state)
                        return f"{func.id.replace('c_', '', 1)}<{parsed_type}>({unwrap_paren(__dfs_stmt(val))})"
                    case "c_short" | "c_ushort" | "c_int" | "c_uint" | "c_long" | "c_ulong" | "c_longlong" | "c_ulonglong":
                        if len(node.args) != 1: raise SyntaxError(f"{func.id} requires exactly one argument")
                        arg = node.args[0]
                        if type(arg) == ast.Constant:
                            value = arg.value
                            if type(value) == int or type(value) == float:
                                val_int = int(value)
                                match func.id:
                                    case "c_short": return str(val_int % (1<<16) - (1<<16) if val_int >= (1<<15) else val_int % (1<<16))
                                    case "c_ushort": return str(val_int % (1<<16))
                                    case "c_int": return str(val_int % (1<<32) - (1<<32) if val_int >= (1<<31) else val_int % (1<<32))
                                    case "c_uint": return str(val_int % (1<<32))
                                    case "c_long": return str(val_int % (1<<32) - (1<<32) if val_int >= (1<<31) else val_int % (1<<32)) + "L"
                                    case "c_ulong": return str(val_int % (1<<32)) + "UL"
                                    case "c_longlong": return str(val_int % (1<<64) - (1<<64) if val_int >= (1<<63) else val_int % (1<<64)) + "LL"
                                    case "c_ulonglong": return str(val_int % (1<<64)) + "ULL"
                        arg_type = eval_type(arg, type_ctx, state, path)
                        if func.id == "c_long" and arg_type.is_py_object:
                            return "PyLong_AsLong(" + unwrap_paren(__dfs_stmt(arg)) + ")"
                        raise NotImplementedError(f"Unsupported argument type for {func.id}(): {type(arg)}")
                    case "c_bool":
                        if len(node.args) != 1: raise SyntaxError("c_bool requires exactly one argument")
                        arg = node.args[0]
                        arg_type = eval_type(arg, type_ctx, state, path)
                        if arg_type.is_py_object:
                            return "(PyObject_IsTrue(" + unwrap_paren(__dfs_stmt(arg)) + ") != 0)"
                        match arg_type.type_:
                            case "builtins.bool":
                                return unwrap_paren(__dfs_stmt(arg))
                            case "builtins.int":
                                return "(" + unwrap_paren(__dfs_stmt(arg)) + " != 0)"
                            case _:
                                raise NotImplementedError(f"Unsupported argument type for c_bool(): {arg_type.type_}")
                    case "c_transform_par_unseq":
                        if len(node.args) != 3: raise SyntaxError("c_transform_par_unseq requires exactly three arguments")
                        src = node.args[0]
                        tgt = node.args[1]
                        func_arg = node.args[2]
                        state.used_builtins.add(Execution)
                        return state.get_name("transform") + "(std::execution::par_unseq, {src}.begin(), {src}.end(), {tgt}.begin(), {func})".format(
                            src=unwrap_paren(__dfs_stmt(src)),
                            tgt=unwrap_paren(__dfs_stmt(tgt)),
                            func=unwrap_paren(__dfs_stmt(func_arg))
                        )
                    case "py_object":
                        if len(node.args) != 1: raise SyntaxError("py_object requires exactly one argument")
                        arg = node.args[0]
                        type_data = eval_type(arg, type_ctx, state, path)
                        if type_data.type_.startswith("py2cpp"):
                            match type_data.type_.split(".")[-1]:
                                case "c_short" | "c_ushort" | "c_int" | "c_uint" | "c_long" | "c_ulong" | "c_longlong" | "c_ulonglong":
                                    return "PyLong_FromLongLong(" + unwrap_paren(__dfs_stmt(arg)) + ")"
                                case "c_float" | "c_double":
                                    return "PyFloat_FromDouble(" + unwrap_paren(__dfs_stmt(arg)) + ")"
                                case "c_char":
                                    return "PyBytes_FromStringAndSize(&" + unwrap_paren(__dfs_stmt(arg)) + ", 1)"
                                case "c_bool":
                                    return "PyBool_FromLong(" + unwrap_paren(__dfs_stmt(arg)) + ")"
                                case "c_void":
                                    return "Py_None"
                                case _:
                                    raise NotImplementedError(f"Unsupported argument type for py_object(): {type_data.type_}")
                        
                        match type_data.type_:
                            case "builtins.str":
                                return "PyUnicode_FromString(" + unwrap_paren(__dfs_stmt(arg)) + ")"
                            case "builtins.int":
                                return "PyLong_FromLongLong(" + unwrap_paren(__dfs_stmt(arg)) + ")"
                            case "builtins.float":
                                return "PyFloat_FromDouble(" + unwrap_paren(__dfs_stmt(arg)) + ")"
                            case "builtins.bool":
                                return "PyBool_FromLong(" + unwrap_paren(__dfs_stmt(arg)) + ")"
                            case _:
                                raise NotImplementedError(f"Unsupported argument type for py_object(): {type_data.type_}") 
                    case "print":
                        keywords = node.keywords

                        kw_end = state.get_name("endl")
                        kw_sep = "\" \""

                        for kwd in keywords:
                            if kwd.arg == "end":
                                kw_end = __dfs_stmt(kwd.value) + " << " + stmt_state.state.get_name("flush")
                            if kwd.arg == "sep":
                                kw_sep = __dfs_stmt(kwd.value)
                        return state.get_name("cout") + " << " + (" << " + kw_sep + " << ").join(__dfs_stmt(arg) for arg in node.args) + " << " + kw_end
                    case "input":
                        kwargs = {"var":direct.value} if direct.type_ == "assign" else {}
                        if len(node.args) == 0:
                            return ScriptTemplateInput.format(stmt_state, **kwargs)
                        return ScriptTemplateInput.format(stmt_state, prompt=__dfs_stmt(node.args[0]), **kwargs)
                    case "int":
                        if len(node.args) == 0:
                            return "0"
                        target = node.args[0]
                        target_type = eval_type(target, type_ctx, state, path)
                        
                        match target_type.type_:
                            case "builtins.int":
                                return unwrap_paren(__dfs_stmt(node.args[0]))
                            case "builtins.float":
                                return "static_cast<long>(" + unwrap_paren(__dfs_stmt(node.args[0])) + ")"
                            case "builtins.bool":
                                return "(" + unwrap_paren(__dfs_stmt(node.args[0])) + " ? 1 : 0)"
                            case "builtins.str":
                                return "std::stol(" + unwrap_paren(__dfs_stmt(node.args[0])) + ")"
                            case _:
                                raise NotImplementedError(f"Unsupported type for int(): {target_type.type_}")
                        raise RuntimeError()
                    case "float":
                        if len(node.args) == 0:
                            return "0.0"
                        target = node.args[0]
                        target_type = eval_type(target, type_ctx, state, path)
                        
                        match target_type.type_:
                            case "builtins.int":
                                return "static_cast<double>(" + unwrap_paren(__dfs_stmt(node.args[0])) + ")"
                            case "builtins.float":
                                return unwrap_paren(__dfs_stmt(node.args[0]))
                            case "builtins.bool":
                                return "(" + unwrap_paren(__dfs_stmt(node.args[0])) + " ? 1.0 : 0.0)"
                            case "builtins.str":
                                return "std::stod(" + unwrap_paren(__dfs_stmt(node.args[0])) + ")"
                            case _:
                                raise NotImplementedError(f"Unsupported type for float(): {target_type.type_}")
                    case "abs":
                        if len(node.args) != 1: raise SyntaxError("abs requires exactly one argument")
                        arg = node.args[0]
                        arg_type = eval_type(arg, type_ctx, state, path)
                        match arg_type.type_:
                            case "builtins.int":
                                return "std::abs(" + unwrap_paren(__dfs_stmt(arg)) + ")"
                            case "builtins.float":
                                state.used_builtins.add(CMath)
                                return "std::fabs(" + unwrap_paren(__dfs_stmt(arg)) + ")"
                            case _:
                                raise NotImplementedError(f"Unsupported type for abs(): {arg_type.type_}")
                    case "len":
                        if len(node.args) != 1: raise SyntaxError("len requires exactly one argument")
                        arg = node.args[0]
                        arg_type = eval_type(arg, type_ctx, state, path)
                        match arg_type.type_:
                            case "builtins.list":
                                return unwrap_paren(__dfs_stmt(arg)) + ".size()"
                            case _:
                                raise NotImplementedError(f"Unsupported type for len(): {arg_type.type_}")
                    case "c_vector_resize":
                        if len(node.args) != 2: raise SyntaxError("c_vector_resize requires exactly two arguments")
                        vector = node.args[0]
                        size = node.args[1]
                        return unwrap_paren(__dfs_stmt(vector)) + ".resize(" + unwrap_paren(__dfs_stmt(size)) + ")"
                    case "divmod":
                        if len(node.args) != 2: raise SyntaxError("divmod requires exactly two arguments")
                        dividend = unwrap_paren(__dfs_stmt(node.args[0]))
                        divisor = unwrap_paren(__dfs_stmt(node.args[1]))
                        return "std::make_tuple(" + dividend + " / " + divisor + ", " + dividend + " % " + divisor + ")"
                    case "exit":
                        state.used_builtins.add(CStdLib)
                        return "std::exit(" + unwrap_paren(__dfs_stmt(node.args[0])) + ")"
                    case "min" | "max":
                        state.used_builtins.add(Algorithm)
                        op = "std::min" if func.id == "min" else "std::max";
                        if len(node.args) == 0:
                            raise SyntaxError(f"{func.id} requires at least one argument")
                        if len(node.args) == 1:
                            arg = node.args[0]
                            arg_str = unwrap_paren(__dfs_stmt(arg))
                            elem_op = "std::min_element" if func.id == "min" else "std::max_element"
                            return f"(*{elem_op}({arg_str}.begin(), {arg_str}.end()))"
                        return f"{op}({{" + ", ".join(unwrap_paren(__dfs_stmt(a)) for a in node.args) + "})"
                    case "sum":
                        state.used_builtins.add(Numeric)
                        if len(node.args) == 0:
                            raise SyntaxError("sum requires at least one argument")
                        iterable = unwrap_paren(__dfs_stmt(node.args[0]))
                        start = unwrap_paren(__dfs_stmt(node.args[1])) if len(node.args) >= 2 else "0"
                        return f"std::accumulate({iterable}.begin(), {iterable}.end(), static_cast<decltype({iterable})::value_type>({start}))"
                    case "sorted":
                        state.used_builtins.add(Algorithm)
                        if len(node.args) != 1: raise SyntaxError("sorted requires exactly one argument")
                        reverse = False
                        for kwd in node.keywords:
                            if kwd.arg == "reverse":
                                if type(kwd.value) == ast.Constant and kwd.value.value is True:
                                    reverse = True
                        tpl = Template_Sorted_Reverse if reverse else Template_Sorted
                        return tpl.format(stmt_state, iterable=unwrap_paren(__dfs_stmt(node.args[0])))
                    case "reversed":
                        state.used_builtins.add(Algorithm)
                        if len(node.args) != 1: raise SyntaxError("reversed requires exactly one argument")
                        return Template_Reversed.format(stmt_state, iterable=unwrap_paren(__dfs_stmt(node.args[0])))
                    case "any" | "all":
                        state.used_builtins.add(Algorithm)
                        op = "std::any_of" if func.id == "any" else "std::all_of"
                        if len(node.args) != 1: raise SyntaxError(f"{func.id} requires exactly one argument")
                        iterable = unwrap_paren(__dfs_stmt(node.args[0]))
                        tmp = state.get_tempid()
                        stmt_state.extra_codes_pre.append(f"const auto& {tmp} = {iterable};")
                        return f"{op}({tmp}.begin(), {tmp}.end(), [](auto __x){{ return static_cast<bool>(__x); }})"
                    case "round":
                        state.used_builtins.add(CMath)
                        if len(node.args) == 1:
                            return f"static_cast<long>(std::round({unwrap_paren(__dfs_stmt(node.args[0]))}))"
                        if len(node.args) == 2:
                            return Template_Round_NDigits.format(stmt_state,
                                x=unwrap_paren(__dfs_stmt(node.args[0])),
                                ndigits=unwrap_paren(__dfs_stmt(node.args[1])))
                        raise SyntaxError("round requires 1 or 2 arguments")
                    case "chr":
                        state.used_builtins.add(String)
                        if len(node.args) != 1: raise SyntaxError("chr requires exactly one argument")
                        return f"std::string(1, static_cast<char>({unwrap_paren(__dfs_stmt(node.args[0]))}))"
                    case "ord":
                        if len(node.args) != 1: raise SyntaxError("ord requires exactly one argument")
                        return f"static_cast<long>(({unwrap_paren(__dfs_stmt(node.args[0]))})[0])"
                    case "open":
                        return __emit_file_open(node)
                    case "c_array":
                        return "{}"
                    case _:
                        if (func.id in state.imports_from_random
                            and not isinstance(type_ctx.type_dict.get(func.id), FunctionTypeData)):
                            return __emit_random_call(state.imports_from_random[func.id], node.args)
                        if (func.id in state.imports_from_math
                            and not isinstance(type_ctx.type_dict.get(func.id), FunctionTypeData)):
                            return __emit_math_call(state.imports_from_math[func.id], node.args)
                        if (func.id in state.imports_from_time
                            and not isinstance(type_ctx.type_dict.get(func.id), FunctionTypeData)):
                            return __emit_time_call(state.imports_from_time[func.id], node.args)

                        typ = eval_type(func, type_ctx, state, path).type_
                        if typ.startswith("py2cpp") and typ.endswith("py_object"):
                            args_list = "PyTuple_Pack(" + str(len(node.args))
                            for arg in node.args:
                                args_list += ", " + unwrap_paren(__dfs_stmt(arg))
                            args_list += ")"
                            return Template_Python_Call.format(
                                stmt_state,
                                func=__dfs_stmt(func),
                                args=args_list
                            )
                        return __dfs_stmt(node.func) + "(" + ", ".join(unwrap_paren(__dfs_stmt(arg)) for arg in node.args) + ")"

            if type(func) == ast.Attribute:
                if (type(func.value) == ast.Name
                    and func.value.id in state.imports_random_module
                    and func.attr in ("randint", "randrange", "shuffle")):
                    return __emit_random_call(func.attr, node.args)
                if (type(func.value) == ast.Name
                    and func.value.id in state.imports_math_module
                    and func.attr in MATH_FUNCS):
                    return __emit_math_call(func.attr, node.args)
                if (type(func.value) == ast.Name
                    and func.value.id in state.imports_time_module
                    and func.attr in TIME_FUNCS):
                    return __emit_time_call(func.attr, node.args)
                if type(func.value) == ast.Name and func.value.id in state.file_vars:
                    fvar = state.file_vars[func.value.id]
                    match func.attr:
                        case "read":
                            state.used_builtins.add(SStream)
                            state.used_builtins.add(String)
                            return Template_FileRead.format(stmt_state, file=fvar)
                        case "readline":
                            state.used_builtins.add(String)
                            return Template_FileReadline.format(stmt_state, file=fvar)
                        case "write":
                            if len(node.args) != 1: raise SyntaxError("file.write requires 1 argument")
                            return f"{fvar} << {unwrap_paren(__dfs_stmt(node.args[0]))}"
                        case "close":
                            return f"{fvar}.close()"
                        case _:
                            raise NotImplementedError(f"Unsupported file method: {func.attr}")

                target_type = eval_type(func.value, type_ctx, state, path)

                if target_type.type_ == "builtins.str":
                    match func.attr:
                        case "index":
                            state.used_builtins.add(Algorithm)
                            return __dfs_stmt(func.value) + ".find(" + ", ".join(unwrap_paren(__dfs_stmt(arg)) for arg in node.args) + ")"
                        case "find":
                            state.used_builtins.add(Str_Find)
                            return "py2c::str::find(" + __dfs_stmt(func.value) + ", " + ", ".join(unwrap_paren(__dfs_stmt(arg)) for arg in node.args) + ")"
                        case "upper":
                            return Template_String_Upper.format(stmt_state, string=__dfs_stmt(func.value))
                        case "lower":
                            return Template_String_Lower.format(stmt_state, string=__dfs_stmt(func.value))
                        case "split":
                            state.used_builtins.add(List)
                            state.used_builtins.add(String)
                            if len(node.args) == 0:
                                state.used_builtins.add(SStream)
                                return Template_StrSplitWs.format(stmt_state, string=__dfs_stmt(func.value))
                            return Template_StrSplit.format(stmt_state,
                                string=__dfs_stmt(func.value),
                                sep=unwrap_paren(__dfs_stmt(node.args[0])))
                        case "join":
                            if len(node.args) != 1: raise SyntaxError("str.join requires 1 argument")
                            state.used_builtins.add(String)
                            return Template_StrJoin.format(stmt_state,
                                sep=__dfs_stmt(func.value),
                                iterable=unwrap_paren(__dfs_stmt(node.args[0])))
                        case "replace":
                            if len(node.args) != 2: raise SyntaxError("str.replace requires 2 arguments")
                            state.used_builtins.add(String)
                            return Template_StrReplace.format(stmt_state,
                                string=__dfs_stmt(func.value),
                                old=unwrap_paren(__dfs_stmt(node.args[0])),
                                new_=unwrap_paren(__dfs_stmt(node.args[1])))
                        case "strip":
                            state.used_builtins.add(String)
                            return Template_StrStrip.format(stmt_state, string=__dfs_stmt(func.value))
                        case "lstrip":
                            state.used_builtins.add(String)
                            return Template_StrLStrip.format(stmt_state, string=__dfs_stmt(func.value))
                        case "rstrip":
                            state.used_builtins.add(String)
                            return Template_StrRStrip.format(stmt_state, string=__dfs_stmt(func.value))
                        case "startswith":
                            if len(node.args) != 1: raise SyntaxError("str.startswith requires 1 argument")
                            state.used_builtins.add(String)
                            s_tmp = state.get_tempid()
                            p_tmp = state.get_tempid()
                            stmt_state.extra_codes_pre.append(f"const std::string& {s_tmp} = {__dfs_stmt(func.value)};")
                            stmt_state.extra_codes_pre.append(f"const std::string& {p_tmp} = {unwrap_paren(__dfs_stmt(node.args[0]))};")
                            return f"({s_tmp}.size() >= {p_tmp}.size() && {s_tmp}.compare(0, {p_tmp}.size(), {p_tmp}) == 0)"
                        case "endswith":
                            if len(node.args) != 1: raise SyntaxError("str.endswith requires 1 argument")
                            state.used_builtins.add(String)
                            s_tmp = state.get_tempid()
                            p_tmp = state.get_tempid()
                            stmt_state.extra_codes_pre.append(f"const std::string& {s_tmp} = {__dfs_stmt(func.value)};")
                            stmt_state.extra_codes_pre.append(f"const std::string& {p_tmp} = {unwrap_paren(__dfs_stmt(node.args[0]))};")
                            return f"({s_tmp}.size() >= {p_tmp}.size() && {s_tmp}.compare({s_tmp}.size() - {p_tmp}.size(), {p_tmp}.size(), {p_tmp}) == 0)"
                        case "count":
                            if len(node.args) != 1: raise SyntaxError("str.count requires 1 argument")
                            return Template_StrCount.format(stmt_state,
                                string=__dfs_stmt(func.value),
                                sub=unwrap_paren(__dfs_stmt(node.args[0])))
                        case "isdigit" | "isalpha" | "isalnum" | "isspace" | "isupper" | "islower":
                            state.used_builtins.add(Algorithm)
                            state.used_builtins.add(CCtype)
                            state.used_builtins.add(String)
                            check_map = {"isdigit": "::isdigit", "isalpha": "::isalpha", "isalnum": "::isalnum",
                                         "isspace": "::isspace", "isupper": "::isupper", "islower": "::islower"}
                            s_tmp = state.get_tempid()
                            stmt_state.extra_codes_pre.append(f"const std::string& {s_tmp} = {__dfs_stmt(func.value)};")
                            return f"(!{s_tmp}.empty() && std::all_of({s_tmp}.begin(), {s_tmp}.end(), [](unsigned char __c){{ return {check_map[func.attr]}(__c) != 0; }}))"
                        case _:
                            raise NotImplementedError(f"Unsupported string method: {func.attr}")

                if target_type.type_.startswith("builtins.dict"):
                    match func.attr:
                        case "get":
                            if len(node.args) < 1: raise SyntaxError("dict.get requires at least 1 argument")
                            tmp = state.get_tempid()
                            d_str = __dfs_stmt(func.value)
                            k_str = unwrap_paren(__dfs_stmt(node.args[0]))
                            default_str = unwrap_paren(__dfs_stmt(node.args[1])) if len(node.args) >= 2 else f"typename std::decay<decltype({d_str})>::type::mapped_type{{}}"
                            stmt_state.extra_codes_pre.append(f"auto {tmp} = {d_str}.find({k_str});")
                            return f"({tmp} != {d_str}.end() ? {tmp}->second : {default_str})"
                        case "keys" | "values" | "items":
                            raise NotImplementedError(f"dict.{func.attr}() should be used inside a for-loop")
                        case "pop":
                            if len(node.args) != 1: raise SyntaxError("dict.pop requires 1 argument")
                            d_str = __dfs_stmt(func.value)
                            k_str = unwrap_paren(__dfs_stmt(node.args[0]))
                            tmp_it = state.get_tempid()
                            tmp_val = state.get_tempid()
                            stmt_state.extra_codes_pre.append(f"auto {tmp_it} = {d_str}.find({k_str});")
                            stmt_state.extra_codes_pre.append(f"auto {tmp_val} = {tmp_it}->second;")
                            stmt_state.extra_codes_pre.append(f"{d_str}.erase({tmp_it});")
                            return tmp_val
                        case "clear":
                            return f"{__dfs_stmt(func.value)}.clear()"
                        case _:
                            raise NotImplementedError(f"Unsupported dict method: {func.attr}")

                if target_type.type_.startswith("builtins.set"):
                    match func.attr:
                        case "add":
                            if len(node.args) != 1: raise SyntaxError("set.add requires 1 argument")
                            return f"{__dfs_stmt(func.value)}.insert({unwrap_paren(__dfs_stmt(node.args[0]))})"
                        case "remove" | "discard":
                            if len(node.args) != 1: raise SyntaxError(f"set.{func.attr} requires 1 argument")
                            return f"{__dfs_stmt(func.value)}.erase({unwrap_paren(__dfs_stmt(node.args[0]))})"
                        case "clear":
                            return f"{__dfs_stmt(func.value)}.clear()"
                        case _:
                            raise NotImplementedError(f"Unsupported set method: {func.attr}")

                if target_type.type_ == "builtins.list":
                    match func.attr:
                        case "append":
                            return __dfs_stmt(func.value) + ".push_back(" + ", ".join(unwrap_paren(__dfs_stmt(arg)) for arg in node.args) + ")"
                        case "pop":
                            if len(node.args) == 0:
                                return Template_Pop_NoIndex.format(stmt_state, list=__dfs_stmt(func.value))
                            return Template_Pop_WithIndex.format(stmt_state, list=__dfs_stmt(func.value), index=__dfs_stmt(node.args[0]))
                        case "clear":
                            return __dfs_stmt(func.value) + ".clear()"
                        case "index":
                            state.used_builtins.add(Algorithm)
                            return f"std::distance({__dfs_stmt(func.value)}.begin(), std::find({__dfs_stmt(func.value)}.begin(), {__dfs_stmt(func.value)}.end(), {', '.join(unwrap_paren(__dfs_stmt(arg)) for arg in node.args)}))"
                        case "sort":
                            state.used_builtins.add(Algorithm)
                            return f"std::sort({__dfs_stmt(func.value)}.begin(), {__dfs_stmt(func.value)}.end())"
                        case "reverse":
                            state.used_builtins.add(Algorithm)
                            return f"std::reverse({__dfs_stmt(func.value)}.begin(), {__dfs_stmt(func.value)}.end())"
                        case "count":
                            state.used_builtins.add(Algorithm)
                            return f"std::count({__dfs_stmt(func.value)}.begin(), {__dfs_stmt(func.value)}.end(), {', '.join(unwrap_paren(__dfs_stmt(arg)) for arg in node.args)})"
                        case "extend":
                            return f"{__dfs_stmt(func.value)}.insert({__dfs_stmt(func.value)}.end(), {__dfs_stmt(node.args[0])}.begin(), {__dfs_stmt(node.args[0])}.end())"
                        case "insert":
                            return f"{__dfs_stmt(func.value)}.insert({__dfs_stmt(func.value)}.begin() + {__dfs_stmt(node.args[0])}, {unwrap_paren(__dfs_stmt(node.args[1]))})"
                        case _:
                            raise NotImplementedError(f"Unsupported list method: {func.attr}")
                return __dfs_stmt(func.value) + "." + func.attr + "(" + ", ".join(unwrap_paren(__dfs_stmt(arg)) for arg in node.args) + ")"

        case ast.Name:
            assert type(node) == ast.Name
            if (node.id in state.imports_from_math
                and state.imports_from_math[node.id] in MATH_CONSTS
                and node.id not in state.defined):
                return __emit_math_const(state.imports_from_math[node.id])
            return node.id

        case ast.BinOp:
            assert type(node) == ast.BinOp
            left_type = eval_type(node.left, type_ctx, state, path)
            right_type = eval_type(node.right, type_ctx, state, path)
            
            assert left_type.is_py_object == right_type.is_py_object, "Mixing py_object with other types is not supported in binary operations"
            
            if left_type.is_py_object:
                py_ops: dict[type[ast.operator], str] = {
                    ast.Add: "PyNumber_Add",
                    ast.Sub: "PyNumber_Subtract",
                    ast.Mult: "PyNumber_Multiply",
                    ast.Div: "PyNumber_TrueDivide",
                    ast.Mod: "PyNumber_Remainder",
                    ast.LShift: "PyNumber_Lshift",
                    ast.RShift: "PyNumber_Rshift",
                    ast.BitOr: "PyNumber_Or",
                    ast.BitAnd: "PyNumber_And",
                    ast.BitXor: "PyNumber_Xor",
                    ast.Pow: "PyNumber_Power"
                }
                if type(node.op) in py_ops:
                    func_name = py_ops[type(node.op)]
                    stmt_state.state.used_builtins.add(Python)
                    return f"{func_name}({__dfs_stmt(node.left)}, {__dfs_stmt(node.right)})"
                raise NotImplementedError(f"Unsupported operator for py_object: {type(node.op)}")
            
            if type(node.op) in (ast.Add, ast.Sub):
                try:
                    
                    if left_type.type_ in ("builtins.int", "builtins.float") and type(node.right) == ast.Constant and isinstance(node.right.value, complex):
                        assert node.right.value.real == 0.0
                        return state.get_name("complex") + "<double>(" + __dfs_stmt(node.left) + ", " + ("" if type(node.op) == ast.Add else "-") + str(abs(node.right.value.imag)) + ")"
                    if right_type.type_ in ("builtins.int", "builtins.float") and type(node.left) == ast.Constant and isinstance(node.left.value, complex):
                        assert node.left.value.real == 0.0
                        return state.get_name("complex") + "<double>(" + (str(abs(node.left.value.imag)) + ", " + ("" if type(node.op) == ast.Add else "-") + str(abs(node.left.value.imag)) + ", ") + __dfs_stmt(node.right) + ")"
                except NotImplementedError:
                    pass
            if type(node.op) == ast.Pow:
                return state.get_name("pow") + "(" + __dfs_stmt(node.left) + ", " + __dfs_stmt(node.right) + ")"
            
            ops: dict[type[ast.operator], str] = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
                ast.Mod: "%",
                ast.LShift: "<<",
                ast.RShift: ">>",
                ast.BitOr: "|",
                ast.BitAnd: "&",
                ast.BitXor: "^"
            }
            return "(" + __dfs_stmt(node.left) + " " + ops[type(node.op)] + " " + __dfs_stmt(node.right) + ")"

        case ast.Subscript:
            assert type(node) == ast.Subscript
            value_type = eval_type(node.value, type_ctx, state, path)
            if value_type.type_ == "tuple":
                return state.get_name("get") + "<" + unwrap_paren(__dfs_stmt(node.slice)) + ">(" + __dfs_stmt(node.value) + ")"

            if type(node.slice) == ast.Slice:
                if node.slice.step is not None:
                    raise NotImplementedError("Slicing with step is not supported")
                tmp = state.get_tempid()
                target_str = __dfs_stmt(node.value)
                stmt_state.extra_codes_pre.append(f"const auto& {tmp} = {target_str};")

                def _slice_idx(expr: ast.expr | None, default: str) -> str:
                    if expr is None:
                        return default
                    s = unwrap_paren(__dfs_stmt(expr))
                    return f"(({s}) < 0 ? static_cast<long>({tmp}.size()) + ({s}) : ({s}))"

                lo = _slice_idx(node.slice.lower, "0")
                hi = _slice_idx(node.slice.upper, f"static_cast<long>({tmp}.size())")

                if value_type.type_ == "builtins.str":
                    return f"{tmp}.substr({lo}, ({hi}) - ({lo}))"
                if value_type.type_.startswith("builtins.list"):
                    elem_type = value_type.generics[0]
                    assert type(elem_type) == TypeData
                    return f"std::vector<{parse_type(elem_type, type_ctx, state)}>({tmp}.begin() + ({lo}), {tmp}.begin() + ({hi}))"
                raise NotImplementedError(f"Slicing not supported for type: {value_type.type_}")

            if (type(node.slice) == ast.UnaryOp and type(node.slice.op) == ast.USub
                and type(node.slice.operand) == ast.Constant and type(node.slice.operand.value) == int):
                tmp = state.get_tempid()
                target_str = __dfs_stmt(node.value)
                stmt_state.extra_codes_pre.append(f"const auto& {tmp} = {target_str};")
                offset = node.slice.operand.value
                return f"{tmp}[{tmp}.size() - {offset}]"

            return __dfs_stmt(node.value) + "[" + unwrap_paren(__dfs_stmt(node.slice)) + "]"

        case ast.UnaryOp:
            assert type(node) == ast.UnaryOp
            
            node_type = eval_type(node.operand, type_ctx, state, path)
            
            if node_type.is_py_object:
                py_unary_ops: dict[type[ast.unaryop], str] = {
                    ast.UAdd: "PyNumber_Positive",
                    ast.USub: "PyNumber_Negative",
                    ast.Not: "PyObject_Not",
                    ast.Invert: "PyNumber_Invert",
                }
                if type(node.op) in py_unary_ops:
                    func_name = py_unary_ops[type(node.op)]
                    stmt_state.state.used_builtins.add(Python)
                    return f"{func_name}({__dfs_stmt(node.operand)})"
                raise NotImplementedError(f"Unsupported unary operator for py_object: {type(node.op)}")
            
            unary_ops: dict[type[ast.unaryop], str] = {
                ast.UAdd: "+",
                ast.USub: "-",
                ast.Not: "!",
                ast.Invert: "~",
            }
            return unary_ops[type(node.op)] + __dfs_stmt(node.operand)

        case ast.Compare:
            assert type(node) == ast.Compare
            comp_ops: dict[type[ast.cmpop], str] = {
                ast.Eq: "==",
                ast.NotEq: "!=",
                ast.Lt: "<",
                ast.LtE: "<=",
                ast.Gt: ">",
                ast.GtE: ">=",
            }
            in_ops = (ast.In, ast.NotIn)
            if len(node.ops) == 1 and type(node.ops[0]) in in_ops:
                container_type = eval_type(node.comparators[0], type_ctx, state, path)
                left_str = unwrap_paren(__dfs_stmt(node.left))
                right_str = __dfs_stmt(node.comparators[0])
                negate = type(node.ops[0]) == ast.NotIn
                if container_type.type_ == "builtins.str":
                    state.used_builtins.add(String)
                    c_tmp = state.get_tempid()
                    stmt_state.extra_codes_pre.append(f"const std::string& {c_tmp} = {right_str};")
                    expr = f"({c_tmp}.find({left_str}) != std::string::npos)"
                elif container_type.type_.startswith("builtins.dict") or container_type.type_.startswith("builtins.set"):
                    expr = f"({right_str}.count({left_str}) > 0)"
                else:
                    state.used_builtins.add(Algorithm)
                    expr = f"(std::find({right_str}.begin(), {right_str}.end(), {left_str}) != {right_str}.end())"
                return f"(!{expr})" if negate else expr
            if len(node.ops) == 1:
                return "(" + __dfs_stmt(node.left) + " " + comp_ops[type(node.ops[0])] + " " + __dfs_stmt(node.comparators[0]) + ")"
            parts: list[str] = []
            prev = __dfs_stmt(node.left)
            for op, comp in zip(node.ops, node.comparators):
                cur = __dfs_stmt(comp)
                parts.append(f"({prev} {comp_ops[type(op)]} {cur})")
                prev = cur
            return "(" + " && ".join(parts) + ")"

        case ast.BoolOp:
            assert type(node) == ast.BoolOp
            bool_ops: dict[type[ast.boolop], str] = {
                ast.And: "&&",
                ast.Or: "||",
            }
            return "(" + (" " + bool_ops[type(node.op)] + " ").join(unwrap_paren(__dfs_stmt(value)) for value in node.values) + ")"

        case ast.IfExp:
            assert type(node) == ast.IfExp
            body_str = __dfs_stmt(node.body)
            if "?" in body_str:
                body_str = "(" + body_str + ")"
            test_str = __dfs_stmt(node.test)
            if ":" in test_str:
                test_str = "(" + test_str + ")"
            orelse_str = __dfs_stmt(node.orelse)
            if ":" in orelse_str:
                orelse_str = "(" + orelse_str + ")"
            return f"({test_str} ? {body_str} : {orelse_str})"

        case _:
            raise NotImplementedError(f"Unsupported AST node type: {type(node)} {ast.dump(node)}")

    raise NotImplementedError(f"Unsupported AST node type: {type(node)} {ast.dump(node)}")

def dfs(node: ast.Module | ast.stmt, type_ctx: TypeContext, state: State, depth: int = 0, path: str = "") -> str:

    def __pad(code: str) -> str:
        return pad(code, state.setting.indent)
    def __dfs(node: ast.stmt) -> str:
        return dfs(node, type_ctx, state, depth + 1, path)
    def __dfs_stmt(node: ast.expr, *, direct: DirectData = DirectData()) -> tuple[str, str, str]:
        stmt_state = StmtState(state)
        _val = dfs_stmt(node, type_ctx, stmt_state, path, direct=direct)
        return _val, stmt_state.collect_pre_extra, stmt_state.collect_post_extra
    def __parse_type(type_: TypeData) -> str:
        return parse_type(type_, type_ctx, state)

    result: str = ""

    if depth == 0 or type(node) == ast.Module:
        children: list[ast.AST] = list(ast.iter_child_nodes(node))
        children_preload: list[ast.FunctionDef] = []
        children_main: list[ast.AST] = []
        for child in children:
            if isinstance(child, ast.FunctionDef):
                children_preload.append(child)
            else:
                children_main.append(child)

        for child in children_preload:
            result += __dfs(child) + "\n"
        result += "int main() {\n"
        for child in children_main:
            assert isinstance(child, ast.stmt)
            code = __dfs(child)
            if code != "":
                result += __pad(code) + "\n"
        result += state.setting.indent + "return 0;\n}\n"
        return result.strip("\n")

    assert isinstance(node, ast.stmt)
    
    if state.setting.py_to_comment:
        start_line = node.lineno - 1
        end_line = node.end_lineno - 1 if node.end_lineno is not None else start_line
        start_col = node.col_offset
        end_col = node.end_col_offset if node.end_col_offset is not None else len(state.origin_code.code.splitlines()[end_line])

        for child in getattr(node, "body", []) + getattr(node, "orelse", []):
            if not isinstance(child, ast.stmt):
                continue
            
            if child.lineno - 1 < end_line:
                end_line = child.lineno - 1
                end_col = child.col_offset - 1
            elif child.lineno - 1 == end_line and child.col_offset < end_col:
                end_col = child.col_offset - 1
            if end_col < 0:
                end_line -= 1
                end_col = 0
        py_code = indexMultiLine(state.origin_code.code, start_line, start_col, end_line, end_col)
        py_code = py_code.rstrip()
        result += "// " + py_code.replace("\n", "\n// ") + "\n"
    
    comments_prev, comments_now = state.origin_code.popComments(node)
    comments_now_code = (" //" + comments_now.text) if comments_now is not None else ""
    
    result += "".join("//"+comment.text+"\n" for comment in comments_prev).lstrip("\n")
    

    match type(node):
        
        case ast.Pass:
            assert type(node) == ast.Pass
            return comments_now_code
        
        case ast.Assign:
            assert type(node) == ast.Assign

            post = ""

            if len(node.targets) == 1:
                target = node.targets[0]
                target_str: str = ""
                is_direct_access = True
                while True:
                    match type(target):
                        case ast.Attribute:
                            assert type(target) == ast.Attribute
                            is_direct_access = False

                            target_str = "." + target.attr + target_str
                            target_new = target.value
                            assert type(target_new) == ast.Name or type(target_new) == ast.Attribute, "Only simple variable assignment is supported" + str(ast.dump(target))
                            target = target_new
                            continue
                        
                        case ast.Subscript:
                            assert type(target) == ast.Subscript
                            is_direct_access = False
                            
                            val, extra_pre, extra_post = __dfs_stmt(target.slice)
                            if extra_pre != "": result += extra_pre + "\n"
                            if extra_post != "": post = extra_post + "\n" + post

                            target_str = "[" + unwrap_paren(val) + "]" + target_str
                            target_new = target.value
                            assert type(target_new) == ast.Name or type(target_new) == ast.Attribute, "Only simple variable assignment is supported" + str(ast.dump(target))
                            target = target_new
                            continue
                            
                        case ast.Name:
                            target_str = assert_type(target, ast.Name).id + target_str
                            break
                        case _:
                            raise NotImplementedError("Only simple variable assignment is supported " + str(ast.dump(target)))
                    break


                val, extra_pre, extra_post = __dfs_stmt(node.value, direct=DirectData(type_="assign", value=target_str))
                
                if val == "": return extra_pre + comments_now_code
                if extra_pre != "": result += extra_pre + "\n"
                if extra_post != "": post = extra_post + "\n" + post

                loc = path + target_str
                if not is_direct_access or loc in state.defined:
                    result += target_str + " = " + unwrap_paren(val) + ";"
                else:
                    type_data = type_ctx.get_vartype(loc)

                    if type_data.type_.split(".")[-1] == "c_array":
                        assert len(type_data.generics) == 2
                        base_type = __parse_type(assert_type(type_data.generics[0], TypeData))
                        array_size_literal = type_data.generics[1]
                        assert type(array_size_literal) == TypeData and array_size_literal.type_ == "Literal" and len(array_size_literal.generics) == 1
                        array_size_data = array_size_literal.generics[0]
                        assert type(array_size_data) == TypeData and array_size_data.type_.isdigit()
                        result += base_type + " " + target_str + "[" + str(array_size_data.type_) + "] = " + val + ";"
                    else:
                        type_ = __parse_type(type_data)
                        result += type_ + " " + target_str + " = " + unwrap_paren(val) + ";"
                    state.defined[loc] = True

            else:
                targets = node.targets
                for target in targets:
                    assert type(target) == ast.Name, "Only simple variable assignment is supported"
                    target_id = target.id

                    loc = path + target_id
                    if loc not in state.defined:
                        result += __parse_type(type_ctx.get_vartype(loc)) + " " + target_id + ";"
                        state.defined[loc] = True

            result += comments_now_code
            if post: result += "\n" + post.strip("\n")

        case ast.AnnAssign:
            assert type(node) == ast.AnnAssign
            target = node.target
            value = node.value

            assert value is not None, "AnnAssign without value is not supported"
            
            is_const = False
            annotation = node.annotation
            if type(annotation) == ast.Subscript:
                type_target = annotation.value
                if type(type_target) == ast.Name and type_target.id == "Final":
                    is_const = True
            const_str = "const " if is_const else ""

            if type(target) == ast.Name:
                loc = path + target.id
                if loc in state.defined:
                    val, extra_pre, extra_post = __dfs_stmt(value)
                    if extra_pre != "": result += extra_pre + "\n"

                    result += path + " = " + val + ";"
                    if extra_post != "": result += "\n" + extra_post
                else:
                    val, extra_pre, extra_post = __dfs_stmt(value)
                    if extra_pre != "": result += extra_pre + "\n"
                    type_data = type_ctx.get_vartype(loc)
                    if type_data.type_.split(".")[-1] == "c_array":
                        assert len(type_data.generics) == 2
                        base_type = __parse_type(assert_type(type_data.generics[0], TypeData))
                        array_size_literal = type_data.generics[1]
                        assert type(array_size_literal) == TypeData and array_size_literal.type_ == "Literal" and len(array_size_literal.generics) == 1
                        array_size_data = array_size_literal.generics[0]
                        assert type(array_size_data) == TypeData and array_size_data.type_.isdigit()
                        result += f"{const_str}{base_type} {target.id}[{str(array_size_data.type_)}] = {val};"
                    else:
                        type_ = __parse_type(type_data)
                        result += f"{const_str}{type_} {target.id} = {val};"
                    if extra_post != "": result += "\n" + extra_post
                    state.defined[loc] = True

            result += comments_now_code

        case ast.AugAssign:
            assert type(node) == ast.AugAssign

            target = node.target
            if type(target) == ast.Name:
                loc = path + target.id
                _op_to_str: dict[type[ast.operator], str] = {
                    ast.Add: "+=",
                    ast.Sub: "-=",
                    ast.Mult: "*=",
                    ast.Div: "/=",
                    ast.Mod: "%=",
                    ast.LShift: "<<=",
                    ast.RShift: ">>=",
                    ast.BitOr: "|=",
                    ast.BitAnd: "&=",
                    ast.BitXor: "^=",
                    ast.FloorDiv: "//=",
                }
                val, extra_pre, extra_post = __dfs_stmt(node.value)
                if extra_pre != "":
                    result += extra_pre + "\n"
                    
                op = _op_to_str[type(node.op)]

                assert loc in state.defined, "AugAssign to undefined variable is not supported"
                
                if op == "+=" and val == "1":
                    result += "++" + target.id + ";"
                elif op == "-=" and val == "1":
                    result += "--" + target.id + ";"
                else:
                    result += target.id + " " + op + " " + val + ";"
                    
                if extra_post != "": result += "\n" + extra_post

            result += comments_now_code
            
                

        case ast.Expr:
            assert type(node) == ast.Expr
            call_expr = node.value
            if type(call_expr) == ast.Call and type(call_expr.func) == ast.Name and call_expr.func.id == "c_exitcode":
                assert depth == 1, "c_exitcode can only be used in the main function"
                val, extra_pre, extra_post = __dfs_stmt(call_expr.args[0])
                if extra_pre != "": result += extra_pre + "\n"
                result += "return " + val + ";"
            elif type(call_expr) == ast.Call and type(call_expr.func) == ast.Name and call_expr.func.id == "c_include":
                if len(call_expr.args) != 1: raise SyntaxError("c_include requires exactly one argument")
                __arg = call_expr.args[0]
                assert type(__arg) == ast.Constant and type(__arg.value) == str
                assert re.match(r'^<.*>$|^".*"$', __arg.value), "c_include argument must be enclosed in <> or \"\""
                state.used_builtins.add(Builtins(f"#include {__arg.value}", inline=True))
                
                extra_post = ""
            elif type(call_expr) == ast.Call and type(call_expr.func) == ast.Name and call_expr.func.id == "c_do":
                if len(call_expr.args) != 1: raise SyntaxError("c_do requires exactly one argument")
                __arg = call_expr.args[0]
                assert type(__arg) == ast.Constant and type(__arg.value) == str
                result += __arg.value.strip()
                
                extra_post = ""
            else:
                val, extra_pre, extra_post = __dfs_stmt(call_expr)
                if extra_pre != "": result += extra_pre + "\n"
                if val: result += val + ";"

            if extra_post != "": result += "\n" + extra_post
            result += comments_now_code

        case ast.If:
            assert type(node) == ast.If
            val, extra_pre, extra_post = __dfs_stmt(node.test)
            if extra_pre != "": result += extra_pre + "\n"

            result += "if (" + unwrap_paren(val) + ") {\n"
            for stmt in node.body:
                result += __pad(dfs(stmt, type_ctx, state, depth + 1, path)) + "\n"
            result += "}"
            if len(node.orelse) > 0:
                if len(node.orelse) == 1 and type(node.orelse[0]) == ast.If:
                    result += " else " + dfs(node.orelse[0], type_ctx, state, depth, path).strip()
                else:
                    result += " else {\n"
                    for stmt in node.orelse:
                        result += __pad(dfs(stmt, type_ctx, state, depth + 1, path)) + "\n"
                    result += "}"
            
            if extra_post != "": result += "\n" + extra_post

        case ast.For:
            assert type(node) == ast.For
            extra_post = extra_post_start = extra_post_step = extra_post_end = ""

            target = node.target
            iter = node.iter
            body = node.body
            if type(target) == ast.Name:
                if type(iter) == ast.Call and type(iter.func) == ast.Name and iter.func.id == "range":
                    loc = path + target.id

                    if len(iter.args) == 1:
                        val_end, extra_pre, extra_post = __dfs_stmt(iter.args[0])
                        if extra_pre != "": result += extra_pre + "\n"

                        start = "0"
                        end = val_end
                        step = "1"
                    elif len(iter.args) == 2:
                        val_start, extra_pre_start, extra_post_start = __dfs_stmt(iter.args[0])
                        val_end, extra_pre_end, extra_post_end = __dfs_stmt(iter.args[1])
                        if extra_pre_start != "": result += extra_pre_start + "\n"
                        if extra_pre_end != "": result += extra_pre_end + "\n"

                        start = val_start
                        end = val_end
                        step = "1"
                    elif len(iter.args) == 3:
                        val_start, extra_pre_start, extra_post_start = __dfs_stmt(iter.args[0])
                        val_end, extra_pre_end, extra_post_end = __dfs_stmt(iter.args[1])
                        val_step, extra_pre_step, extra_post_step = __dfs_stmt(iter.args[2])
                        if extra_pre_start != "": result += extra_pre_start + "\n"
                        if extra_pre_end != "": result += extra_pre_end + "\n"
                        if extra_pre_step != "": result += extra_pre_step + "\n"

                        start = val_start
                        end = val_end
                        step = val_step
                    else:
                        raise ValueError("Invalid number of arguments for range()")
                    
                    exp_change = target.id + " += " + step
                    if step == "1": exp_change = "++" + target.id
                    if step == "-1": exp_change = "--" + target.id
                    
                    state.used_builtins.add(CStdDef)
                    result += "for (" + ("long long " if loc not in state.defined else "") + target.id + " = " + start + "; " + target.id + " < " + end + "; " + exp_change + ") {\n"
                else:
                    val, extra_pre, extra_post = __dfs_stmt(iter)
                    if extra_pre != "": result += extra_pre + "\n"
                    result += "for (auto " + target.id + " : " + val + ") {\n"
            else:
                raise NotImplementedError("Only simple variable as for loop target is supported")
            type_ctx.type_dict[path + target.id] = TypeData("py2cpp.api.c_longlong")
            for stmt in body:
                result += __pad(dfs(stmt, type_ctx, state, depth + 1, path)) + "\n"
            result += "}"
            
            if extra_post_start != "": result += "\n" + extra_post_start
            if extra_post_step != "": result += "\n" + extra_post_step
            if extra_post_end != "": result += "\n" + extra_post_end
            if extra_post != "": result += "\n" + extra_post

        case ast.While:
            assert type(node) == ast.While

            val, extra_pre, extra_post = __dfs_stmt(node.test)
            if extra_pre != "": result += extra_pre + "\n"

            result += "while (" + unwrap_paren(val) + ") {\n"
            for stmt in node.body:
                result += __pad(dfs(stmt, type_ctx, state, depth + 1, path)) + "\n"
            result += "}"
            
            if extra_post != "": result += "\n" + extra_post

        case ast.FunctionDef:
            assert type(node) == ast.FunctionDef

            loc = path + node.name
            fn_type = assert_type(type_ctx.get_vartype(loc), FunctionTypeData)
            ret_type = fn_type.return_type
            result += __parse_type(ret_type) + " " + node.name + "("
            args = fn_type.args
            if len(args) > 0:
                for i, arg in enumerate(args):
                    arg_name = node.args.args[i].arg
                    arg_type = arg
                    result += __parse_type(arg_type[1]) + " " + arg_name + ", "
            result = result.rstrip(", ") + ") {\n"
            for stmt in node.body:
                newline = dfs(stmt, type_ctx, state, depth + 1, path + node.name + "#")
                if newline != "":
                    result += __pad(newline) + "\n"
            result += "}\n"

        case ast.Return:
            assert type(node) == ast.Return

            value = node.value
            if (type(value) == ast.Name and value.id == "void") or value is None:
                result += "return;"
            else:
                val, extra_pre, extra_post = __dfs_stmt(value)
                if extra_pre != "": result += extra_pre + "\n"
                
                if extra_post == "":
                    result += "return " + unwrap_paren(val) + ";"
                else:
                    retVar = state.get_tempid()
                    result += f"""auto {retVar} = {unwrap_paren(val)};\n{extra_post}\nreturn {retVar};"""

            result += comments_now_code

        case ast.ClassDef:
            assert type(node) == ast.ClassDef

            if any(
                isinstance(dec, ast.Name) and dec.id == 'c_struct'
                for dec in node.decorator_list
            ): result = ""
            else: raise NotImplementedError("Only @c_struct decorator is supported for class definitions")

        case ast.Import:
            assert type(node) == ast.Import
            for alias in node.names:
                top_module = alias.name.split(".")[0]
                if top_module == "random":
                    state.imports_random_module.add(alias.asname or alias.name)
                elif top_module == "math":
                    state.imports_math_module.add(alias.asname or alias.name)
                elif top_module == "time":
                    state.imports_time_module.add(alias.asname or alias.name)
                elif top_module in ("sys", "py2cpp"):
                    pass
                else:
                    raise NotImplementedError("Only sys, py2cpp, random, math, and time are supported")

        case ast.ImportFrom:
            assert type(node) == ast.ImportFrom
            if node.module == "random":
                for alias in node.names:
                    if alias.name in ("randint", "randrange", "shuffle"):
                        local_name = alias.asname or alias.name
                        state.imports_from_random[local_name] = alias.name
            elif node.module == "math":
                for alias in node.names:
                    if alias.name in MATH_FUNCS or alias.name in MATH_CONSTS:
                        local_name = alias.asname or alias.name
                        state.imports_from_math[local_name] = alias.name
            elif node.module == "time":
                for alias in node.names:
                    if alias.name in TIME_FUNCS:
                        local_name = alias.asname or alias.name
                        state.imports_from_time[local_name] = alias.name

        case ast.Continue:
            result += "continue;" + comments_now_code

        case ast.Break:
            result += "break;" + comments_now_code

        case ast.Global:
            assert type(node) == ast.Global
            raise SyntaxError(f"global statement is not supported (file: {state.origin_code}:{node.lineno})")

        case ast.With:
            assert type(node) == ast.With

            assert len(node.items) == 1, "Only single item 'with' statement is supported"
            ctx_expr = node.items[0].context_expr

            if type(ctx_expr) == ast.Call and type(ctx_expr.func) == ast.Name:
                match ctx_expr.func.id:
                    case "c_global":
                        for stmt in node.body:
                            state.global_code += dfs(stmt, type_ctx, state, depth, path) + "\n"
                    case "c_skip":
                        pass
                    case "open":
                        optvar = node.items[0].optional_vars
                        assert type(optvar) == ast.Name, "with open(...) requires 'as <name>'"
                        fname = optvar.id

                        if len(ctx_expr.args) < 1: raise SyntaxError("open requires at least 1 argument")
                        stmt_state_open = StmtState(state)
                        path_str = unwrap_paren(dfs_stmt(ctx_expr.args[0], type_ctx, stmt_state_open, path))
                        mode = "r"
                        if len(ctx_expr.args) >= 2:
                            mode_node = ctx_expr.args[1]
                            if type(mode_node) == ast.Constant and type(mode_node.value) == str:
                                mode = mode_node.value
                        for kwd in ctx_expr.keywords:
                            if kwd.arg == "mode" and type(kwd.value) == ast.Constant and type(kwd.value.value) == str:
                                mode = kwd.value.value
                        state.used_builtins.add(FStream)
                        cls = "std::ofstream" if ("w" in mode or "a" in mode) else "std::ifstream"
                        flags = ""
                        if "a" in mode:
                            flags = ", std::ios::app"
                        if "b" in mode:
                            flags += ", std::ios::binary"

                        actual_name = "__py2c_file_" + state.get_tempid()
                        state.file_vars[fname] = actual_name

                        if stmt_state_open.collect_pre_extra:
                            result += stmt_state_open.collect_pre_extra + "\n"
                        result += f"{cls} {actual_name}({path_str}{flags});\n"
                        for stmt in node.body:
                            line = dfs(stmt, type_ctx, state, depth, path)
                            if line != "":
                                result += line + "\n"
                        result += f"{actual_name}.close();"
                        if stmt_state_open.collect_post_extra:
                            result += "\n" + stmt_state_open.collect_post_extra

                        del state.file_vars[fname]
                    case _:
                        raise NotImplementedError(f"Unsupported 'with' context: {ctx_expr.func.id}")

            else:
                raise NotImplementedError(f"Unsupported 'with' context: {ast.dump(ctx_expr)}")

        case _:
            raise NotImplementedError(f"Unsupported AST node type: {type(node)} {ast.dump(node)}")

    return result

@dataclass
class CppnizeResult:
    code: str
    type_ctx: TypeContext
    state: State

def py_2_cpp(text: str, path: str = "<string>", *, setting: Setting | None = None, verbose: bool = False) -> CppnizeResult:
    compile(text, filename=path, mode='exec', flags=ast.PyCF_ONLY_AST, dont_inherit=True, optimize=-1)
    
    _setting = setting or Setting()

    _parse_types = get_time(parse_types) if verbose else parse_types
    type_ctx = _parse_types(text, force_typed=not _setting.use_py)

    tree = ast.parse(text, filename=path)

    state = State(origin_code=Code(text), setting=_setting)
    code = ""
    _dfs = get_time(dfs) if verbose else dfs
    code_body = _dfs(tree, type_ctx, state=state)



    
    code_struct = ""
    for struct in type_ctx.struct_dict.values():
        code_struct += "struct " + struct.name + " {\n"
        for field_name, field_type in struct.fields.items():
            code_struct += "    " + parse_type(field_type.type_, type_ctx, state) + " " + field_name + ";\n"
        code_struct += "};\n\n"
        



    code_builtin: str = ""
    include_set: set[str] = set()
    for b in state.used_builtins:
        code_builtin += b.toString(include_set)

    if include_set:
        code_builtin += "\n"

    __is_used_builtin: Callable[[str], bool] = lambda name: NameDict[name][0] in state.used_builtins
    for name in state.setting.minimize_namespace:
        builtin, full_name = NameDict[name]
        if builtin in state.used_builtins:
            code_builtin += f"using {full_name};\n"

    if any(map(__is_used_builtin, state.setting.minimize_namespace)) and not code.endswith("\n\n"):
        code_builtin += "\n"


    
    code_global = ""
    state.global_code = state.global_code.strip("\n")
    if state.global_code:
        code_global += state.global_code + "\n\n"

    code = code_builtin + code_struct + code_global + code_body

    return CppnizeResult(code=code, type_ctx=type_ctx, state=state)

def build_cpp_to_exe(cpp_code: str, output_path: str):
    import subprocess
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False) as f:
        f.write(cpp_code)
        tmp_cpp_path = f.name

    try:
        compile_command = [PATH_COMPILER, tmp_cpp_path, "-o", output_path, "-std=c++11"]
        result = subprocess.run(compile_command, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{result.stderr}")
        
    finally:
        Path(tmp_cpp_path).unlink()

def py_2_exe(text: str, path: str, *, setting: Setting | None = None, verbose: bool = False):
    path_target = Path(path)

    code = py_2_cpp(text, path=path, verbose=verbose, setting=setting).code

    path_temp = path_here / "temp"
    path_temp.mkdir(exist_ok=True)

    (path_temp / (path_target.stem + ".cpp")).write_text(code, encoding="utf-8")

    _build_cpp_to_exe = get_time(build_cpp_to_exe) if verbose else build_cpp_to_exe
    _build_cpp_to_exe(code, str(path_temp / (path_target.stem + ".exe")))

    result_name = (path_target.parent) / (path_target.stem + ".exe")
    if result_name.exists():
        result_name.unlink()
    (path_temp / (path_target.stem + ".exe")).rename(result_name)

    shutil.rmtree(path_temp)
