from typing import Callable, TypeVar

T = TypeVar("T")
U = TypeVar("U")

def c_transform_par_unseq(src: list[T], tgt: list[U], fn: Callable[[T], U]) -> None:
    for i in range(len(src)):
        tgt[i] = fn(src[i])