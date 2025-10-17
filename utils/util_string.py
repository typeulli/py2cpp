
def pad(text: str, n: int = 1) -> str:
    return "    " * n + text.replace("\n", "\n" + "    " * n)

def unwrap_paren(s: str) -> str:
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    return s

def indexMultiLine(text: str, startline: int, startcol: int, endline: int, endcol: int) -> str:
    lines = text.splitlines()[startline:endline+1]
    if startline == endline:
        return lines[0][startcol-1:endcol]
    lines[0] = lines[0][startcol-1:]
    lines[-1] = lines[-1][:endcol]
    return "\n".join(lines)