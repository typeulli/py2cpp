
def pad(text: str, n: int = 1) -> str:
    return "    " * n + text.replace("\n", "\n" + "    " * n)

def unwrap_paren(s: str) -> str:
    s = s.strip()
    while True:
        if not (s.startswith("(") and s.endswith(")")):
            break
        depth = 0
        is_str = False
        for i, c in enumerate(s):
            if c == '"' or c == "'":
                is_str = not is_str
            if is_str:
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            if depth == 0 and i != len(s) - 1:
                return s
        s = s[1:-1].strip()
    return s

def indexMultiLine(text: str, startline: int, startcol: int, endline: int, endcol: int) -> str:
    lines = text.splitlines()[startline:endline+1]
    if startline == endline:
        return lines[0][startcol-1:endcol]
    lines[0] = lines[0][startcol-1:]
    lines[-1] = lines[-1][:endcol]
    return "\n".join(lines)