from typing import Optional, TypeAlias
from dataclasses import dataclass
import tempfile
import re
import ast

from mypy import api

GenericData: TypeAlias = "list[TypeData | GenericData]"

@dataclass
class TypeData:
    type_: str
    generics: GenericData
    
    @classmethod
    def from_str(cls, type_str: str) -> "TypeData":
        if type_str.startswith("def "):
            return FunctionTypeData.from_str(type_str)
        
        if "[" not in type_str:
            return cls(type_=type_str, generics=[])
        
        typename = type_str[:type_str.index("[")]
        if not ("[" in type_str and type_str.endswith("]")):
            return cls(type_=type_str, generics=[])
        generics_str = type_str[type_str.index("[")+1:-1]
        generics: GenericData = []
        depth = 0
        current_generic = ""
        for char in generics_str:
            if char == "[":
                depth += 1
                current_generic += char
            elif char == "]":
                depth -= 1
                current_generic += char
            elif char == "," and depth == 0:
                generics.append(TypeData.from_str(current_generic.strip()))
                current_generic = ""
            else:
                current_generic += char
        if current_generic:
            generics.append(TypeData.from_str(current_generic.strip()))
        return cls(type_=typename, generics=generics)

class FunctionTypeData(TypeData):
    def __init__(self, args: list[tuple[str, TypeData]], return_type: TypeData):
        super().__init__(type_="function", generics=[return_type])

        self.args = args
        self.return_type = return_type

    @classmethod
    def from_str(cls, type_str: str) -> "FunctionTypeData":
        _match = re.fullmatch(r"def \((.*)\) -> (.*)", type_str)
        args: list[tuple[str, TypeData]] = []
        if _match:
            ret_type = _match.group(2)

            arg_string = _match.group(1)
            if arg_string != "":
                for arg in _match.group(1).split(", "):
                    _arg_match = re.fullmatch(r"(.*): (.*)", arg)
                    assert _arg_match
                    arg_name = _arg_match.group(1)
                    arg_type = _arg_match.group(2)
                    args.append((arg_name, TypeData.from_str(arg_type)))
            return cls(
                args=args,
                return_type=TypeData.from_str(ret_type)
            )
        else:
            _match_noreturn = re.fullmatch(r"def \((.*)\)", type_str)
            if _match_noreturn:
                arg_string = _match_noreturn.group(1)
                if arg_string != "":
                    for arg in _match_noreturn.group(1).split(", "):
                        _arg_match = re.fullmatch(r"(.*): (.*)", arg)
                        assert _arg_match
                        arg_name = _arg_match.group(1)
                        arg_type = _arg_match.group(2)
                        args.append((arg_name, TypeData.from_str(arg_type)))
                return cls(
                    args=args,
                    return_type=TypeData(type_="builtins.None", generics=[])
                )
                
        raise ValueError("Invalid function type", type_str)

@dataclass
class StructFieldData:
    type_: TypeData
    default: Optional[ast.expr] = None

@dataclass
class StructData:
    name: str
    fields: dict[str, StructFieldData]

class RevealTypeInserter(ast.NodeTransformer):
    def __init__(self) -> None:
        self.scope_stack: list[Optional[str]] = []

        self.revealed_structs: list[StructData] = []

    def visit_Assign(self, node: ast.Assign) -> list[ast.AST]:
        self.generic_visit(node)
        new_nodes: list[ast.AST] = [node]
        for target in node.targets:
            if isinstance(target, ast.Name):
                reveal_call = ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id='reveal_type', ctx=ast.Load()),
                        args=[ast.Name(id=target.id, ctx=ast.Load())],
                        keywords=[]
                    )
                )
                # 전역 변수: scope=None, 함수 내부 변수: scope=현재 함수
                setattr(reveal_call, "scope", self.scope_stack[-1] if self.scope_stack else None)
                setattr(reveal_call, "var_name", target.id)
                new_nodes.append(reveal_call)
        return new_nodes
    
    def visit_AnnAssign(self, node: ast.AnnAssign) -> list[ast.AST]:
        self.generic_visit(node)
        new_nodes: list[ast.AST] = [node]

        if isinstance(node.target, ast.Name):
            reveal_call = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id='reveal_type', ctx=ast.Load()),
                    args=[ast.Name(id=node.target.id, ctx=ast.Load())],
                    keywords=[]
                ),
                lineno=getattr(node.target, "lineno", node.lineno),
                col_offset=0
            )
            setattr(reveal_call, "scope", self.scope_stack[-1] if self.scope_stack else None)
            setattr(reveal_call, "var_name", node.target.id)
            new_nodes.append(reveal_call)

        return new_nodes

    def visit_FunctionDef(self, node: ast.FunctionDef) -> list[ast.AST]:
        self.scope_stack.append(node.name)
        new_body: list[ast.stmt] = []

        # 매개변수 reveal_type 삽입
        for arg in node.args.args:
            reveal_call = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id='reveal_type', ctx=ast.Load()),
                    args=[ast.Name(id=arg.arg, ctx=ast.Load())],
                    keywords=[]
                )
            )
            # AST attribute에 scope 저장
            setattr(reveal_call, "scope", node.name)
            setattr(reveal_call, "var_name", arg.arg)
            new_body.append(reveal_call)

        for stmt in node.body:
            new_body.append(self.visit(stmt))

        node.body = new_body
        self.scope_stack.pop()

        # 함수 이름 itself에 reveal_type
        reveal_func = ast.Expr(
            value=ast.Call(
            func=ast.Name(id='reveal_type', ctx=ast.Load()),
            args=[ast.Name(id=node.name, ctx=ast.Load())],
            keywords=[]
            )
        )
        setattr(reveal_func, "scope", None)
        setattr(reveal_func, "var_name", node.name)

        return [node, reveal_func]
        
    def visit_ClassDef(self, node: ast.ClassDef) -> list[ast.AST]:
        # if not at root level, raise
        if self.scope_stack:
            raise ValueError("Nested class definitions are not supported")
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()
        # 데코레이터 확인: @c_struct가 있는지
        has_c_struct = any(
            isinstance(dec, ast.Name) and dec.id == 'c_struct'
            for dec in node.decorator_list
        )
        
        
        if has_c_struct:
            fields: dict[str, StructFieldData] = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    field_name = stmt.target.id
                    field_type = ast.unparse(stmt.annotation)
                    fields[field_name] = StructFieldData(type_=TypeData.from_str(field_type), default=stmt.value)
                    
            self.revealed_structs.append(StructData(name=node.name, fields=fields))
        return [node]


def find_node_path(tree: ast.AST, target_node: ast.AST) -> Optional[list[ast.AST]]:
    """
    tree: AST root
    target_node: 찾고 싶은 노드
    return: root -> target_node까지의 경로 리스트
    """
    path: list[ast.AST] = []

    def visit(node: ast.AST, current_path: list[ast.AST]) -> bool:
        current_path.append(node)

        if node is target_node:
            path.extend(current_path)
            return True  # 찾았음

        for child in ast.iter_child_nodes(node):
            if visit(child, current_path.copy()):
                return True

        return False

    found = visit(tree, [])
    if found:
        return path
    return None

@dataclass
class TypeContext:
    type_dict: dict[str, TypeData]
    struct_dict: dict[str, StructData]

    def get_vartype(self, key: str) -> TypeData:
        if "." not in key:
            return self.type_dict[key]
        var, *attrs = key.split(".")
        current_type = self.type_dict[var]
        for attr in attrs:
            struct_data = self.struct_dict[current_type.type_.split(".")[-1]]
            field_data = struct_data.fields[attr]
            current_type = field_data.type_
        return current_type

def parse_types(text: str, path_scripts: list[str]) -> TypeContext:

    tree = ast.parse(text)

    transformer = RevealTypeInserter()
    transformer.visit(tree)

    ast.fix_missing_locations(tree)
    
    new_code = ast.unparse(tree)
    new_tree = ast.parse(new_code)
    new_code_lines = new_code.splitlines()
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(new_code)
        tmp_path = f.name

    import os
    os.environ["MYPYPATH"] = os.pathsep.join(path_scripts)
    result = api.run([tmp_path, "--strict", "--show-error-codes", "--no-error-summary"])
    os.remove(tmp_path)


    type_dict: dict[str, TypeData] = {
        "input": FunctionTypeData(
            args=[
                ("prompt", TypeData(type_="builtins.str", generics=[]))
            ], 
            return_type=TypeData(type_="builtins.str", generics=[])),
    }
    
    for struct in transformer.revealed_structs:
        type_dict[struct.name] = TypeData(
            type_= "type",
            generics=[TypeData(type_=struct.name, generics=[])]
        )

    for line in result[0].splitlines():
        if "Revealed type is" in line:
            parts = line.split("note: Revealed type is")
            type_str = parts[1].strip().strip("'").strip('"')
            lineno = int(parts[0].split(".py")[-1].strip().split(":")[1]) - 1
            code_line = new_code_lines[lineno].strip()
            if code_line.startswith("reveal_type("):
                var_name = code_line[len("reveal_type("):-1]

                # 스코프 확인
                # 함수 매개변수: key="func#var", 전역 변수: key="var"
                # AST에서 scope 정보 찾기
                key = var_name
                for node in ast.walk(new_tree):
                    # if node is calling  reveal_type
                    if isinstance(node, ast.Expr):
                        if node.value.lineno-1 == lineno:
                            # get parents path
                            l = find_node_path(new_tree, node)
                            if type(l) == list:
                                s = ""
                                for x in l:
                                    if type(x) == ast.Module:
                                        continue
                                    _name = getattr(x, "name", None)
                                    if _name is not None:
                                        s += _name + "#"
                                key = s + var_name
                            break
                assert type_str != "Any", f"Type of {key} is Any"
                type_dict[key] = TypeData.from_str(type_str)
    
    for struct in transformer.revealed_structs:
        for field_name, field_data in struct.fields.items():
            field_key = struct.name + "#" + field_name
            if field_key in type_dict:
                field_data.type_ = type_dict[field_key]
            else:
                raise ValueError(f"Field type for {field_key} not found in revealed types")

    return TypeContext(type_dict=type_dict, struct_dict={s.name: s for s in transformer.revealed_structs})