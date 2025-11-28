from dataclasses import dataclass
from io import StringIO
import tokenize

@dataclass
class CommentInfo:
    lineno: int
    col: int
    end_lineno: int
    end_col: int
    text: str

def parse_comments(code: str, char: str = "-") -> list[CommentInfo]:
    assert len(char) == 1, "char must be a single character"

    """Remove all code and return only comments (including inline comments)."""
    result_lines: list[CommentInfo] = []
    tokens = tokenize.generate_tokens(StringIO(code).readline)

    for tok_type, tok_str, start, end, _ in tokens:
        if tok_type == tokenize.COMMENT:
            result_lines.append(
                CommentInfo(
                    lineno=start[0], col=start[1],
                    end_lineno=end[0], end_col=end[1],
                    text=tok_str.replace("#", "", 1)
                )
            )
    return result_lines

if __name__ == "__main__":
    print(*parse_comments("""
    def fast_inverse_root(x: float) -> float:
        '''Compute an approximation to the inverse square root of x.'''
        import struct

        threehalfs = 1.5

        x2 = x * 0.5
        y = x

        # Convert float to int bits
        i = struct.unpack('i', struct.pack('f', y))[0]
        # Magic number and bit manipulation
        i = 0x5f3759df - (i >> 1)
        # Convert bits back to float
        y = struct.unpack('f', struct.pack('i', i))[0]

        # One iteration of Newton's method
        y = y * (threehalfs - (x2 * y * y))

        return y  # Approximate inverse square root of x
    """), sep="\n")