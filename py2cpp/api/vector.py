from typing import TypeVar

T = TypeVar("T")

def c_vector_resize(vec: list[T], new_size: int) -> None:
    current_size = len(vec)
    if new_size < current_size:
        del vec[new_size:]
    else:
        vec.extend([None] * (new_size - current_size)) # type: ignore