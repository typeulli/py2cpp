from dataclasses import dataclass, field
from pathlib import Path
import ast
import random
import re
import shutil
import time
import tempfile
from typing import TypeVar, ParamSpec, Callable
from functools import wraps


from utils.util_string import pad, unwrap_paren
from utils.util_type import parse_types, parse_func_type


P = ParamSpec("P")
T = TypeVar("T")
def get_time(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} execution time: {end_time - start_time} seconds")
        return result
    return wrapper

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

String = Builtins("#include <string>", inline=True)
List = Builtins("#include <vector>", inline=True)
Str_Find = Builtins("str/find.cpp", requires=[String])
CCtype = Builtins("#include <cctype>", inline=True)
Algorithm = Builtins("#include <algorithm>", inline=True)
Str_Upper = Builtins("str/upper.cpp", requires=[String, Algorithm, CCtype])
Str_Lower = Builtins("str/lower.cpp", requires=[String, Algorithm, CCtype])
CStdLib = Builtins("#include <cstdlib>", inline=True)
Tuple = Builtins("#include <tuple>", inline=True)

NameDict: dict[str, str] = {
    "string": "std::string",
    "vector": "std::vector",
    "tuple": "std::tuple",
    "cin": "std::cin",
    "cout": "std::cout",
    "endl": "std::endl",
    "getline": "std::getline",
    "get": "std::get",
}

@dataclass
class Setting:
    minimize_namespace: list[str]

@dataclass
class State:
    origin_code: str
    defined: dict[str, bool] = field(default_factory=lambda: dict[str, bool]())
    used_builtins: set[Builtins] = field(default_factory=lambda: set[Builtins]())
    used_tempids: set[str] = field(default_factory=lambda: set[str]())
    setting: Setting = field(default_factory=lambda: Setting(minimize_namespace=[]))
    def get_tempid(self) -> str:
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
        while True:
            id_ = "".join(random.choice(chars) for _ in range(8))
            if id_[0] not in "0123456789" and id_ not in self.used_tempids:
                self.used_tempids.add(id_)
                return id_

    def get_name(self, base: str) -> str:
        if base in self.setting.minimize_namespace:
            return base
        return NameDict[base]

@dataclass
class StmtState:
    state: State
    extra_codes: list[str] = field(default_factory=lambda: [])

    @property
    def collect_extra_codes(self) -> str:
        return "\n".join(self.extra_codes)


@dataclass
class ScriptTemplate:
    code: str
    result_expr: str
    require_tmps: list[str] = field(default_factory=lambda: [])
    require_vars: list[str] = field(default_factory=lambda: [])
    require_names: list[str] = field(default_factory=lambda: [])

    def __post_init__(self):
        self.code = self.code.lstrip("\n").rstrip("\n")
    
    def format(self, stmt_state: StmtState, **kwargs: str) -> str:
        state = stmt_state.state
        __get_tempid = lambda: state.get_tempid()
        
        tmp_vars = {tmp: __get_tempid() for tmp in self.require_tmps}
        for var in self.require_vars:
            assert var in kwargs, f"Missing variable '{var}' for template"
    
        names = {"__"+name: state.get_name(name) for name in self.require_names}

        stmt_state.extra_codes.append(self.code.format(**tmp_vars, **kwargs, **names))
        return self.result_expr.format(**tmp_vars, **kwargs, **names)

Template_Pop_NoIndex = ScriptTemplate(
    code="""
auto {tmp1} = {list};
auto {tmp2} = {tmp1}.back();
{tmp1}.pop_back();
""",
    result_expr="{tmp2}",
    require_tmps=["tmp1", "tmp2"], require_vars=["list"]
)

Template_Pop_WithIndex = ScriptTemplate(
    code="""
auto {tmp1} = {list};
auto {tmp2} = {tmp1}[{index}];
{tmp1}.erase({tmp1}.begin() + {index});
""",
    result_expr="{tmp2}",
    require_tmps=["tmp1", "tmp2"], require_vars=["list", "index"]
)



Template_Input_NoPrompt = ScriptTemplate(
    code="""
{__string} {tmp1};
{__getline}({__cin}, {tmp1});
""",
    result_expr="{tmp1}",
    require_tmps=["tmp1"],
    require_names=["cin", "string", "getline"]
)

Template_Input_WithPrompt = ScriptTemplate(
    code="""
{__cout} << {prompt};
{__string} {tmp1};
{__getline}({__cin}, {tmp1});
""",
    result_expr="{tmp1}",
    require_vars=["prompt"], require_tmps=["tmp1"],
    require_names=["cout", "cin", "string", "getline"]
)

def eval_type(expr: ast.expr, type_dict: dict[str, str], state: State, path: str = "") -> str:
    if isinstance(expr, ast.Name):
        type_ = type_dict[path + expr.id]
        assert type_ != "Any"
        return type_
    if isinstance(expr, ast.Call):
        _, ret_type = parse_func_type(eval_type(expr.func, type_dict, state, path))
        return ret_type
    
    
    raise NotImplementedError(f"Unsupported AST node type for eval_type: {type(expr)} {ast.dump(expr)}")

type_alias = {
    "py2c_header.c_int": "int",
    "py2c_header.c_uint": "unsigned int",
    "py2c_header.c_short": "short",
    "py2c_header.c_ushort": "unsigned short",
    "py2c_header.c_long": "long",
    "py2c_header.c_ulong": "unsigned long",
    "py2c_header.c_longlong": "long long",
    "py2c_header.c_ulonglong": "unsigned long long",
    "py2c_header.c_float": "float",
    "py2c_header.c_double": "double",
    "py2c_header.c_char": "char",
    "py2c_header.c_bool": "bool",
    "builtins.int": "long",
    "builtins.float": "double",
    "builtins.bool": "bool",
}
def parse_type(type_: str, state: State) -> str:
    def __parse_type(type_: str) -> str:
        return parse_type(type_, state)
    
    if type_ == "builtins.str":
        state.used_builtins.add(String)
        return state.get_name("string")
    
    if type_ in type_alias:
        return type_alias[type_]
    
    if type_.startswith("builtins.list"):
        state.used_builtins.add(List)
        return state.get_name("vector") + "<" + __parse_type(type_[len("builtins.list")+1:-1]) + ">"
    
    if type_.startswith("tuple"):
        state.used_builtins.add(Tuple)
        elem_types: list[str] = []
        inner = type_[len("tuple")+1:-1]
        depth = 0
        last_idx = 0
        for i, c in enumerate(inner):
            if c == "," and depth == 0:
                elem_types.append(inner[last_idx:i].strip())
                last_idx = i + 1
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
        elem_types.append(inner[last_idx:].strip())
        return state.get_name("tuple") + "<" + ", ".join(__parse_type(t) for t in elem_types) + ">"
    
    raise ValueError(f"Unknown type: {type_}")

def dfs_stmt(node: ast.AST, type_dict: dict[str, str], stmt_state: StmtState, path: str = "") -> str:
    state = stmt_state.state
    def __dfs_stmt(node: ast.AST) -> str:
        return dfs_stmt(node, type_dict, stmt_state, path)

    match type(node):
        case ast.Constant:
            assert type(node) == ast.Constant
            if type(node.value) == bool:
                return "true" if node.value else "false"
            return state.origin_code.splitlines()[node.lineno - 1][node.col_offset:node.end_col_offset]
        
        case ast.List:
            assert type(node) == ast.List
            return "{" + ", ".join(unwrap_paren(__dfs_stmt(elt)) for elt in node.elts) + "}"
        
        case ast.Tuple:
            assert type(node) == ast.Tuple
            return "std::make_tuple(" + ", ".join(unwrap_paren(__dfs_stmt(elt)) for elt in node.elts) + ")"
        
        case ast.Call:
            assert type(node) == ast.Call
            func = node.func
            if type(func) == ast.Name:
                if func.id == "cast":
                    assert len(node.args) == 2
                    type_str = node.args[0]
                    val = node.args[1]
                    assert type(type_str) == ast.Constant and type(type_str.value) == str
                    parsed_type = parse_type(type_str.value, state)
                    return f"({parsed_type})({unwrap_paren(__dfs_stmt(val))})"
                if func.id == "print":
                    keywords = node.keywords
                    
                    kw_end = state.get_name("endl")
                    kw_sep = "\" \""

                    for kwd in keywords:
                        if kwd.arg == "end":
                            kw_end = __dfs_stmt(kwd.value)
                        if kwd.arg == "sep":
                            kw_sep = __dfs_stmt(kwd.value)
                    return state.get_name("cout") + " << " + (" << " + kw_sep + " << ").join(__dfs_stmt(arg) for arg in node.args) + " << " + kw_end
                if func.id == "input":
                    if len(node.args) == 0:
                        return Template_Input_NoPrompt.format(stmt_state)
                    return Template_Input_WithPrompt.format(stmt_state, prompt=__dfs_stmt(node.args[0]))
                if func.id == "int":
                    return "std::stol(" + unwrap_paren(__dfs_stmt(node.args[0])) + ")"
                if func.id == "float":
                    return "std::stod(" + unwrap_paren(__dfs_stmt(node.args[0])) + ")"
                if func.id == "exit":
                    state.used_builtins.add(CStdLib)
                    return "std::exit(" + unwrap_paren(__dfs_stmt(node.args[0])) + ")"
                return __dfs_stmt(node.func) + "(" + ", ".join(unwrap_paren(__dfs_stmt(arg)) for arg in node.args) + ")"

            if type(func) == ast.Attribute:
                target_type = eval_type(func.value, type_dict, state, path)

                if target_type == "builtins.str":
                    match func.attr:
                        case "index":
                            state.used_builtins.add(Algorithm)
                            return __dfs_stmt(func.value) + ".find(" + ", ".join(unwrap_paren(__dfs_stmt(arg)) for arg in node.args) + ")"
                        case "find":
                            state.used_builtins.add(Str_Find)
                            return "py2c::str::find(" + __dfs_stmt(func.value) + ", " + ", ".join(unwrap_paren(__dfs_stmt(arg)) for arg in node.args) + ")"
                        case "upper":
                            state.used_builtins.add(Str_Upper)
                            return "py2c::str::upper(" + __dfs_stmt(func.value) + ")"
                        case "lower":
                            state.used_builtins.add(Str_Lower)
                            return "py2c::str::lower(" + __dfs_stmt(func.value) + ")"
                        case _:
                            raise NotImplementedError(f"Unsupported string method: {func.attr}")

                if target_type.startswith("builtins.list"):
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
            return node.id

        case ast.BinOp:
            assert type(node) == ast.BinOp
            
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
            if eval_type(node.value, type_dict, state, path).startswith("tuple"):
                return state.get_name("get") + "<" + unwrap_paren(__dfs_stmt(node.slice)) + ">(" + __dfs_stmt(node.value) + ")"
            return __dfs_stmt(node.value) + "[" + unwrap_paren(__dfs_stmt(node.slice)) + "]"

        case ast.UnaryOp:
            assert type(node) == ast.UnaryOp
            unary_ops: dict[type[ast.unaryop], str] = {
                ast.UAdd: "+",
                ast.USub: "-",
                ast.Not: "!",
                ast.Invert: "~",
            }
            return unary_ops[type(node.op)] + __dfs_stmt(node.operand)

        case ast.Compare:
            assert type(node) == ast.Compare
            assert len(node.ops) == 1 and len(node.comparators) == 1
            comp_ops: dict[type[ast.cmpop], str] = {
                ast.Eq: "==",
                ast.NotEq: "!=",
                ast.Lt: "<",
                ast.LtE: "<=",
                ast.Gt: ">",
                ast.GtE: ">=",
            }
            return "(" + __dfs_stmt(node.left) + " " + comp_ops[type(node.ops[0])] + " " + __dfs_stmt(node.comparators[0]) + ")"

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
            return test_str + " ? " + body_str + " : " + orelse_str

        case _:
            raise NotImplementedError(f"Unsupported AST node type: {type(node)} {ast.dump(node)}")

    raise RuntimeError(f"Unsupported AST node type: {type(node)} {ast.dump(node)}")

def dfs(node: ast.AST, type_dict: dict[str, str], state: State, depth: int = 0, path: str = "") -> str:

    def __dfs(node: ast.AST) -> str:
        return dfs(node, type_dict, state, depth + 1, path)
    def __dfs_stmt(node: ast.AST) -> tuple[str, str]:
        stmt_state = StmtState(state)
        _val = dfs_stmt(node, type_dict, stmt_state, path)
        return _val, stmt_state.collect_extra_codes
    def __parse_type(type_: str) -> str:
        return parse_type(type_, state)
    
    
    result: str = ""
    
    if depth == 0:
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
            code = __dfs(child)
            if code != "":
                result += pad(code) + "\n"
        result += "    return 0;\n}\n"
        return result.strip("\n")

        
    match type(node):
        case ast.Assign:
            assert type(node) == ast.Assign
            
            if len(node.targets) == 1:
                target = node.targets[0]
                assert type(target) == ast.Name, "Only simple variable assignment is supported"
                target_id = target.id
                
                loc = path + target_id
                if loc in state.defined:
                    val, extra = __dfs_stmt(node.value)
                    if extra != "": result += extra + "\n"

                    result += target_id + " = " + val + ";"
                else:
                    val, extra = __dfs_stmt(node.value)
                    if extra != "": result += extra + "\n"

                    result += __parse_type(type_dict[loc]) + " " + target_id + " = " + val + ";"
                    state.defined[loc] = True
                    
            else:
                targets = node.targets
                for target in targets:
                    assert type(target) == ast.Name, "Only simple variable assignment is supported"
                    target_id = target.id

                    loc = path + target_id
                    if loc not in state.defined:
                        result += __parse_type(type_dict[loc]) + " " + target_id + ";"
                        state.defined[loc] = True

        case ast.AnnAssign:
            assert type(node) == ast.AnnAssign
            target = node.target
            value = node.value
            
            assert value is not None, "AnnAssign without value is not supported"
            
            if type(target) == ast.Name:
                loc = path + target.id
                if loc in state.defined:
                    val, extra = __dfs_stmt(value)
                    if extra != "": result += extra + "\n"
                    
                    result += path + " = " + val + ";"
                else:
                    val, extra = __dfs_stmt(value)
                    if extra != "": result += extra + "\n"
                    result += __parse_type(type_dict[loc]) + " " + target.id + " = " + val + ";"
                    state.defined[loc] = True
        
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
                val, extra = __dfs_stmt(node.value)
                if extra != "":
                    result += extra + "\n"
                    
                    
                if loc in state.defined:
                    result += target.id + " " + _op_to_str[type(node.op)] + " " + val + ";"
                else:
                    result += __parse_type(type_dict[loc]) + " " + target.id + " " + _op_to_str[type(node.op)] + " " + val + ";"
                    state.defined[loc] = True

        case ast.Expr:
            assert type(node) == ast.Expr
            val, extra = __dfs_stmt(node.value)
            if extra != "": result += extra + "\n"
            result += val + ";"
        case ast.If:
            assert type(node) == ast.If
            val, extra = __dfs_stmt(node.test)
            if extra != "": result += extra + "\n"


            result += "if (" + unwrap_paren(val) + ") {\n"
            for stmt in node.body:
                result += pad(dfs(stmt, type_dict, state, depth + 1, path)) + "\n"
            result += "}"
            if len(node.orelse) > 0:
                if len(node.orelse) == 1 and type(node.orelse[0]) == ast.If:
                    result += " else " + dfs(node.orelse[0], type_dict, state, depth, path).strip()
                else:
                    result += " else {\n"
                    for stmt in node.orelse:
                        result += pad(dfs(stmt, type_dict, state, depth + 1, path)) + "\n"
                    result += "}"
        case ast.For:
            assert type(node) == ast.For
            
            target = node.target
            iter = node.iter
            body = node.body
            if type(target) == ast.Name:
                if type(iter) == ast.Call and type(iter.func) == ast.Name and iter.func.id == "range":
                    loc = path + target.id
                    
                    if len(iter.args) == 1:
                        val_end, extra = __dfs_stmt(iter.args[0])
                        if extra != "": result += extra + "\n"
                        
                        
                        start = "0"
                        end = val_end
                        step = "1"
                    elif len(iter.args) == 2:
                        val_start, extra_start = __dfs_stmt(iter.args[0])
                        val_end, extra_end = __dfs_stmt(iter.args[1])
                        if extra_start != "": result += extra_start + "\n"
                        if extra_end != "": result += extra_end + "\n"

                        start = val_start
                        end = val_end
                        step = "1"
                    elif len(iter.args) == 3:
                        val_start, extra_start = __dfs_stmt(iter.args[0])
                        val_end, extra_end = __dfs_stmt(iter.args[1])
                        val_step, extra_step = __dfs_stmt(iter.args[2])
                        if extra_start != "": result += extra_start + "\n"
                        if extra_end != "": result += extra_end + "\n"
                        if extra_step != "": result += extra_step + "\n"

                        start = val_start
                        end = val_end
                        step = val_step
                    else:
                        raise ValueError("Invalid number of arguments for range()")
                    result += "for (" + ("long " if loc not in state.defined else "") + target.id + " = " + start + "; " + target.id + " < " + end + "; " + target.id + " += " + step + ") {\n"
                else:
                    val, extra = __dfs_stmt(iter)
                    if extra != "": result += extra + "\n"
                    result += "for (auto " + target.id + " : " + val + ") {\n"
            else:
                raise NotImplementedError("Only simple variable as for loop target is supported")
            for stmt in body:
                result += pad(dfs(stmt, type_dict, state, depth + 1, path)) + "\n"
            result += "}"

        case ast.While:
            assert type(node) == ast.While
            
            val, extra = __dfs_stmt(node.test)
            if extra != "": result += extra + "\n"

            result += "while (" + unwrap_paren(val) + ") {\n"
            for stmt in node.body:
                result += pad(dfs(stmt, type_dict, state, depth + 1, path)) + "\n"
            result += "}"

        case ast.FunctionDef:
            assert type(node) == ast.FunctionDef
            loc = path + node.name
            fn_type = type_dict[loc]
            _match = re.fullmatch(r"def \((.*)\) -> (.*)", fn_type)
            if _match:
                ret_type = _match.group(2)
                result += __parse_type(ret_type) + " " + node.name + "("
                args = _match.group(1).split(", ")
                if args != ['']:
                    for arg in args:
                        _arg_match = re.fullmatch(r"(.*): (.*)", arg)
                        assert _arg_match
                        arg_name = _arg_match.group(1)
                        arg_type = _arg_match.group(2)
                        result += __parse_type(arg_type) + " " + arg_name + ", "
                result = result.rstrip(", ") + ") {\n"
                for stmt in node.body:
                    result += pad(dfs(stmt, type_dict, state, depth + 1, path + node.name + "#")) + "\n"
                result += "}\n"
            else:
                _match_noreturn = re.fullmatch(r"def \((.*)\)", fn_type)
                if _match_noreturn:
                    result += "void " + node.name + "("
                    args = _match_noreturn.group(1).split(", ")
                    if args != ['']:
                        for arg in args:
                            _arg_match = re.fullmatch(r"(.*): (.*)", arg)
                            assert _arg_match
                            arg_name = _arg_match.group(1)
                            arg_type = _arg_match.group(2)
                            result += __parse_type(arg_type) + " " + arg_name + ", "
                    result = result.rstrip(", ") + ") {\n"
                    for stmt in node.body:
                        result += pad(dfs(stmt, type_dict, state, depth + 1, path + node.name + "#")) + "\n"
                    result += "}\n"
                else:
                    raise ValueError(f"Invalid function type: {fn_type}")

        case ast.Return:
            assert type(node) == ast.Return
            value = node.value
            if value is None:
                result += "return;"
            else:
                val, extra = __dfs_stmt(value)
                if extra != "": result += extra + "\n"
                result += "return " + unwrap_paren(val) + ";"
        case ast.Import:
            assert type(node) == ast.Import
            if len(node.names) == 1 and node.names[0].name in ("sys", "py2cpp_header"):
                pass
            else:
                raise NotImplementedError("Only 'import py2cpp_header' is supported")
        case ast.ImportFrom:
            assert type(node) == ast.ImportFrom
            if node.module == "py2cpp_header":
                pass
        case ast.Continue:
            result += "continue;"
        case ast.Break:
            result += "break;"
        case ast.Global:
            assert type(node) == ast.Global
            raise SyntaxError(f"global statement is not supported (file: {state.origin_code}:{node.lineno})")
        case _:
            raise NotImplementedError(f"Unsupported AST node type: {type(node)} {ast.dump(node)}")
    return result



def py_2_cpp(text: str, path: str = "<string>", *, setting: Setting | None = None, verbose: bool = False) -> str:

    _parse_types = get_time(parse_types) if verbose else parse_types
    type_dict = _parse_types(text, str(path_here))

    tree = ast.parse(text, filename=path)
    
    state = State(origin_code=text, setting=setting or Setting(minimize_namespace=[]))
    code = "#include <iostream>\n"
    _dfs = get_time(dfs) if verbose else dfs
    code_body = _dfs(tree, type_dict, state=state)

    include_set: set[str] = set()
    for b in state.used_builtins:
        code += b.toString(include_set)
        
    code += "\n"

    for name in state.setting.minimize_namespace:
        code += "using " + NameDict[name] + ";\n"

    code += "\n" + code_body
    
    return code

def build_cpp_to_exe(cpp_code: str, output_path: str):
    import subprocess
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False) as f:
        f.write(cpp_code)
        tmp_cpp_path = f.name

    compile_command = ["g++", tmp_cpp_path, "-o", output_path, "-std=c++11"]

    result = subprocess.run(compile_command, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed: {result.stderr}")
    return

def py_2_exe(text: str, path: str, *, setting: Setting | None = None, verbose: bool = False):
    path_target = Path(path)

    code = py_2_cpp(text, path=path, verbose=verbose, setting=setting)

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

if __name__ == "__main__":
    path_target = path_here / "test" / "main.py"
    d = py_2_exe(path_target.read_text(), path=str(path_target), verbose=True, setting=Setting(minimize_namespace=["cout", "cin", "endl", "get", "string"]))