from functools import cache
from typing import Optional
import tempfile
import re
import ast

from mypy import api

class RevealTypeInserter(ast.NodeTransformer):
    def __init__(self) -> None:
        self.scope_stack: list[Optional[str]] = []

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

@cache
def parse_func_type(type_str: str) -> tuple[list[tuple[str, str]], str]:
    _match = re.fullmatch(r"def \((.*)\) -> (.*)", type_str)
    if _match:
        ret_type = _match.group(2)
        args: list[tuple[str, str]] = []
        
        arg_string = _match.group(1)
        if arg_string != "":
            for arg in _match.group(1).split(", "):
                _arg_match = re.fullmatch(r"(.*): (.*)", arg)
                assert _arg_match
                arg_name = _arg_match.group(1)
                arg_type = _arg_match.group(2)
                args.append((arg_name, arg_type))
        return args, ret_type
    raise ValueError("Invalid function type")

def parse_types(text: str) -> dict[str, str]:

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

    # mypy 실행
    import os
    os.environ["MYPYPATH"] = "C:\\Users\\USER\\Desktop\\py2c\\"
    result = api.run([tmp_path, "--strict", "--show-error-codes", "--no-error-summary"])


    type_dict = {
        "input": "def (prompt: builtins.str) -> builtins.str",
    }

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
                type_dict[key] = type_str
    return type_dict