from typing import TypeVar

T = TypeVar("T")
def assert_type(value: object, type_: type[T]) -> T:
    assert isinstance(value, type_), f"Expected type {type_.__name__}, got {value!r}"
    return value