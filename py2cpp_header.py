from typing import Callable, Generic, NoReturn, Protocol, SupportsIndex, TypeAlias, TypeVar, Optional, Type, overload
from typing import cast, Literal, Final
from types import TracebackType

T = TypeVar("T")
N = TypeVar("N", bound=int)

c_cast = cast
c_static_cast = cast

class c_void: ...
void: Final[c_void] = c_void()

def c_exitcode(code: int) -> NoReturn:
    exit(code)

class c_global:
    def __enter__(self) -> "c_global":
        return self
    def __exit__(self, _exc_type: Optional[Type[BaseException]], _exc_value: Optional[BaseException], _traceback: Optional[TracebackType]) -> None:
        return None

class c_skip:
    def __enter__(self) -> "c_skip":
        return self
    def __exit__(self, _exc_type: Optional[Type[BaseException]], _exc_value: Optional[BaseException], _traceback: Optional[TracebackType]) -> None:
        return None

def c_struct(cls: type[T]) -> type[T]:
    return cls

class c_array(list[T], Generic[T, N]):
    def __init__(self, length: int):
        super().__init__([cast(T, None)] * length)
    def append(self, _: T) -> None: raise NotImplementedError("append is not supported for c_array")
    def remove(self, _: T) -> None: raise NotImplementedError("remove is not supported for c_array")
    def insert(self, index: SupportsIndex, _: T) -> None: raise NotImplementedError("insert is not supported for c_array")
    def pop(self, index: SupportsIndex = 0) -> T: raise NotImplementedError("pop is not supported for c_array")

ctype_int: TypeAlias = "c_int | c_uint | c_short | c_ushort | c_long | c_ulong | c_longlong | c_ulonglong"
clike_int: TypeAlias = "int | ctype_int"
T_ctype_int = TypeVar("T_ctype_int", "c_int", "c_uint", "c_short", "c_ushort", "c_long", "c_ulong", "c_longlong", "c_ulonglong")


class c_integer(Protocol):
    @classmethod
    def cast(cls, x: int) -> int: ...

class c_short:
    def __init__(self, value: int): self.value = value
    def __int__(self) -> int: return self.value

class c_ushort:
    def __init__(self, value: int): self.value = value
    def __int__(self) -> int: return self.value




class c_int:
    def __init__(self, value: int): self.value = value
    def __int__(self) -> int: return self.value

class c_uint:
    def __init__(self, value: int): self.value = value
    def __int__(self) -> int: return self.value


class clike_long:
    INT32_MAX = (1<<31) - 1
    INT32_MIN = -(1<<31)
    UINT32_MAX = 1<<32


_c_casted_long: TypeAlias = "int | c_int | c_uint | c_short | c_ushort | c_long"
class c_long(c_integer, clike_long):
    
    @classmethod
    def cast(cls, x: int) -> int:
        x = x % cls.UINT32_MAX
        if x > cls.INT32_MAX:
            x -= cls.UINT32_MAX
        return x
    
    @overload
    def _arith(self, other: "c_ulong", op: Callable[[int, int], int]) -> "c_ulong": ...
    @overload
    def _arith(self, other: "c_longlong", op: Callable[[int, int], int]) -> "c_longlong": ...
    @overload
    def _arith(self, other: "c_ulonglong", op: Callable[[int, int], int]) -> "c_ulonglong": ...
    @overload
    def _arith(self, other: _c_casted_long, op: Callable[[int, int], int]) -> "c_long": ...
    def _arith(self, other: clike_int, op: Callable[[int, int], int]) -> "c_long | c_ulong | c_longlong | c_ulonglong":
        res = op(self.value, int(other))
        if isinstance(other, c_ulong):
            return c_ulong(res)
        if isinstance(other, c_longlong):
            return c_longlong(res)
        if isinstance(other, c_ulonglong):
            return c_ulonglong(res)
        return c_long(res)

    def __init__(self, value: int): self.value = self.cast(value)
    def __int__(self) -> int: return self.value


    @overload
    def __add__(self, other: "c_ulong") -> "c_ulong": ...
    @overload
    def __add__(self, other: "c_longlong") -> "c_longlong": ...
    @overload
    def __add__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    def __add__(self, other: clike_int) -> "c_long | c_ulong | c_longlong | c_ulonglong":
        return self._arith(other, int.__add__)
    
    
    @overload
    def __sub__(self, other: "c_ulong") -> "c_ulong": ...
    @overload
    def __sub__(self, other: "c_longlong") -> "c_longlong": ...
    @overload
    def __sub__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    def __sub__(self, other: clike_int) -> "c_long | c_ulong | c_longlong | c_ulonglong":
        return self._arith(other, int.__sub__)
    
    
    @overload
    def __mul__(self, other: "c_ulong") -> "c_ulong": ...
    @overload
    def __mul__(self, other: "c_longlong") -> "c_longlong": ...
    @overload
    def __mul__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    def __mul__(self, other: clike_int) -> "c_long | c_ulong | c_longlong | c_ulonglong":
        return self._arith(other, int.__mul__)


    @overload
    def __floordiv__(self, other: "c_ulong") -> "c_ulong": ...
    @overload
    def __floordiv__(self, other: "c_longlong") -> "c_longlong": ...
    @overload
    def __floordiv__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    def __floordiv__(self, other: clike_int) -> "c_long | c_ulong | c_longlong | c_ulonglong":
        return self._arith(other, int.__floordiv__)
    
    
    @overload
    def __truediv__(self, other: "c_ulong") -> "c_ulong": ...
    @overload
    def __truediv__(self, other: "c_longlong") -> "c_longlong": ...
    @overload
    def __truediv__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    def __truediv__(self, other: clike_int) -> "c_long | c_ulong | c_longlong | c_ulonglong":
        return self._arith(other, int.__floordiv__)


    @overload
    def __mod__(self, other: "c_ulong") -> "c_ulong": ...
    @overload
    def __mod__(self, other: "c_longlong") -> "c_longlong": ...
    @overload
    def __mod__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    def __mod__(self, other: clike_int) -> "c_long | c_ulong | c_longlong | c_ulonglong":
        return self._arith(other, int.__mod__)


    @overload
    def __pow__(self, other: "c_ulong") -> "c_ulong": ...
    @overload
    def __pow__(self, other: "c_longlong") -> "c_longlong": ...
    @overload
    def __pow__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    def __pow__(self, other: clike_int) -> "c_long | c_ulong | c_longlong | c_ulonglong":
        return self._arith(other, int.__pow__)


    def __lshift__(self, other: clike_int) -> "c_long":
        return c_long(self.value << int(other))
    def __rshift__(self, other: clike_int) -> "c_long":
        return c_long(self.value >> int(other))


    @overload
    def __and__(self, other: "c_ulong") -> "c_ulong": ...
    @overload
    def __and__(self, other: "c_longlong") -> "c_longlong": ...
    @overload
    def __and__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    def __and__(self, other: clike_int) -> "c_long | c_ulong | c_longlong | c_ulonglong":
        return self._arith(other, int.__and__)


    @overload
    def __or__(self, other: "c_ulong") -> "c_ulong": ...
    @overload
    def __or__(self, other: "c_longlong") -> "c_longlong": ...
    @overload
    def __or__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    def __or__(self, other: clike_int) -> "c_long | c_ulong | c_longlong | c_ulonglong":
        return self._arith(other, int.__or__)


    @overload
    def __xor__(self, other: "c_ulong") -> "c_ulong": ...
    @overload
    def __xor__(self, other: "c_longlong") -> "c_longlong": ...
    @overload
    def __xor__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    def __xor__(self, other: clike_int) -> "c_long | c_ulong | c_longlong | c_ulonglong":
        return self._arith(other, int.__xor__)


    def __neg__(self) -> "c_longlong":
        return c_longlong(-self.value)
    def __pos__(self) -> "c_longlong":
        return c_longlong(+self.value)
    def __invert__(self) -> "c_longlong":
        return c_longlong(~self.value)
    def __str__(self) -> str:
        return str(self.value)
    def __repr__(self) -> str:
        return str(self.value)

    
    
_c_casted_ulong: TypeAlias = "int | c_int | c_uint | c_short | c_ushort | c_long | c_ulong"
class c_ulong(c_integer, clike_long):

    @classmethod
    def cast(cls, x: int) -> int:
        return x % cls.UINT32_MAX
    
    @overload
    def _arith(self, other: "c_longlong", op: Callable[[int, int], int]) -> "c_longlong": ...
    @overload
    def _arith(self, other: "c_ulonglong", op: Callable[[int, int], int]) -> "c_ulonglong": ...
    @overload
    def _arith(self, other: _c_casted_ulong, op: Callable[[int, int], int]) -> "c_ulong": ...
    def _arith(self, other: clike_int, op: Callable[[int, int], int]) -> "c_ulong | c_longlong | c_ulonglong":
        res = op(self.value, int(other))
        if isinstance(other, c_longlong):
            return c_longlong(res)
        if isinstance(other, c_ulonglong):
            return c_ulonglong(res)
        return c_ulong(res)

    
    def __init__(self, value: int): self.value = self.cast(value)
    def __int__(self) -> int: return self.value
    
    
    @overload
    def __add__(self, other: "c_longlong") -> "c_longlong": ...
    @overload
    def __add__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    @overload
    def __add__(self, other: _c_casted_ulong) -> "c_ulong": ...
    def __add__(self, other: clike_int) -> "c_ulong | c_longlong | c_ulonglong":
        return self._arith(other, int.__add__)
    
    
    @overload
    def __sub__(self, other: "c_longlong") -> "c_longlong": ...
    @overload
    def __sub__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    @overload
    def __sub__(self, other: _c_casted_ulong) -> "c_ulong": ...
    def __sub__(self, other: clike_int) -> "c_ulong | c_longlong | c_ulonglong":
        return self._arith(other, int.__sub__)
    
    
    @overload
    def __mul__(self, other: "c_longlong") -> "c_longlong": ...
    @overload
    def __mul__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    @overload
    def __mul__(self, other: _c_casted_ulong) -> "c_ulong": ...
    def __mul__(self, other: clike_int) -> "c_ulong | c_longlong | c_ulonglong":
        return self._arith(other, int.__mul__)
    
    
    @overload
    def __floordiv__(self, other: "c_longlong") -> "c_longlong": ...
    @overload
    def __floordiv__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    @overload
    def __floordiv__(self, other: _c_casted_ulong) -> "c_ulong": ...
    def __floordiv__(self, other: clike_int) -> "c_ulong | c_longlong | c_ulonglong":
        return self._arith(other, int.__floordiv__)
    
    
    @overload
    def __truediv__(self, other: "c_longlong") -> "c_longlong": ...
    @overload
    def __truediv__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    @overload
    def __truediv__(self, other: _c_casted_ulong) -> "c_ulong": ...
    def __truediv__(self, other: clike_int) -> "c_ulong | c_longlong | c_ulonglong":
        return self._arith(other, int.__floordiv__)
    
    
    @overload
    def __mod__(self, other: "c_longlong") -> "c_longlong": ...
    @overload
    def __mod__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    @overload
    def __mod__(self, other: _c_casted_ulong) -> "c_ulong": ...
    def __mod__(self, other: clike_int) -> "c_ulong | c_longlong | c_ulonglong":
        return self._arith(other, int.__mod__)
    
    
    @overload
    def __pow__(self, other: "c_longlong") -> "c_longlong": ...
    @overload
    def __pow__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    @overload
    def __pow__(self, other: _c_casted_ulong) -> "c_ulong": ...
    def __pow__(self, other: clike_int) -> "c_ulong | c_longlong | c_ulonglong":
        return self._arith(other, int.__pow__)
    
    
    def __lshift__(self, other: clike_int) -> "c_ulong":
        return c_ulong(self.value << int(other))
    def __rshift__(self, other: clike_int) -> "c_ulong":
        return c_ulong(self.value >> int(other))
    
    
    @overload
    def __and__(self, other: "c_longlong") -> "c_longlong": ...
    @overload
    def __and__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    @overload
    def __and__(self, other: _c_casted_ulong) -> "c_ulong": ...
    def __and__(self, other: clike_int) -> "c_ulong | c_longlong | c_ulonglong":
        return self._arith(other, int.__and__)
    
    
    @overload
    def __or__(self, other: "c_longlong") -> "c_longlong": ...
    @overload
    def __or__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    @overload
    def __or__(self, other: _c_casted_ulong) -> "c_ulong": ...
    def __or__(self, other: clike_int) -> "c_ulong | c_longlong | c_ulonglong":
        return self._arith(other, int.__or__)
    
    
    @overload
    def __xor__(self, other: "c_longlong") -> "c_longlong": ...
    @overload
    def __xor__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    @overload
    def __xor__(self, other: _c_casted_ulong) -> "c_ulong": ...
    def __xor__(self, other: clike_int) -> "c_ulong | c_longlong | c_ulonglong":
        return self._arith(other, int.__xor__)
    
    
    def __neg__(self) -> "c_longlong":
        return c_longlong(-self.value)
    def __pos__(self) -> "c_longlong":
        return c_longlong(+self.value)
    def __invert__(self) -> "c_longlong":
        return c_longlong(~self.value)
    def __str__(self) -> str:
        return str(self.value)
    def __repr__(self) -> str:
        return str(self.value)

class clike_longlong:
    INT64_MAX = (1<<63) - 1
    INT64_MIN = -(1<<63)
    UINT64_MAX = 1<<64

_c_casted_longlong: TypeAlias = "int | c_int | c_uint | c_short | c_ushort | c_long | c_ulong | c_longlong"
class c_longlong(c_integer, clike_longlong):

    @classmethod
    def cast(cls, x: int) -> int:
        x = x % cls.UINT64_MAX
        if x > cls.INT64_MAX:
            x -= cls.UINT64_MAX
        return x

    @overload
    def _arith(self, other: "c_ulonglong", op: Callable[[int, int], int]) -> "c_ulonglong": ...
    @overload
    def _arith(self, other: _c_casted_longlong, op: Callable[[int, int], int]) -> "c_longlong": ...
    def _arith(self, other: clike_int, op: Callable[[int, int], int]) -> "c_longlong | c_ulonglong":
        res = op(self.value, int(other))
        if isinstance(other, c_ulonglong):
            return c_ulonglong(res)
        return c_longlong(res)


    def __init__(self, value: int): self.value = self.cast(value)
    def __int__(self) -> int: return self.value
    
    
    @overload
    def __add__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    @overload
    def __add__(self, other: _c_casted_longlong) -> "c_longlong": ...
    def __add__(self, other: clike_int) -> "c_longlong | c_ulonglong":
        return self._arith(other, int.__add__)
    
    
    @overload
    def __sub__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    @overload
    def __sub__(self, other: _c_casted_longlong) -> "c_longlong": ...
    def __sub__(self, other: clike_int) -> "c_longlong | c_ulonglong":
        return self._arith(other, int.__sub__)
    
    
    @overload
    def __mul__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    @overload
    def __mul__(self, other: _c_casted_longlong) -> "c_longlong": ...
    def __mul__(self, other: clike_int) -> "c_longlong | c_ulonglong":
        return self._arith(other, int.__mul__)


    @overload
    def __floordiv__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    @overload
    def __floordiv__(self, other: _c_casted_longlong) -> "c_longlong": ...
    def __floordiv__(self, other: clike_int) -> "c_longlong | c_ulonglong":
        return self._arith(other, int.__floordiv__)


    @overload
    def __truediv__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    @overload
    def __truediv__(self, other: _c_casted_longlong) -> "c_longlong": ...
    def __truediv__(self, other: clike_int) -> "c_longlong | c_ulonglong":
        return self._arith(other, int.__floordiv__)


    @overload
    def __mod__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    @overload
    def __mod__(self, other: _c_casted_longlong) -> "c_longlong": ...
    def __mod__(self, other: clike_int) -> "c_longlong | c_ulonglong":
        return self._arith(other, int.__mod__)


    @overload
    def __pow__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    @overload
    def __pow__(self, other: _c_casted_longlong) -> "c_longlong": ...
    def __pow__(self, other: clike_int) -> "c_longlong | c_ulonglong":
        return self._arith(other, int.__pow__)


    def __lshift__(self, other: clike_int) -> "c_longlong":
        return c_longlong(self.value << int(other))
    def __rshift__(self, other: clike_int) -> "c_longlong":
        return c_longlong(self.value >> int(other))


    @overload
    def __and__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    @overload
    def __and__(self, other: _c_casted_longlong) -> "c_longlong": ...
    def __and__(self, other: clike_int) -> "c_longlong | c_ulonglong":
        return self._arith(other, int.__and__)


    @overload
    def __or__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    @overload
    def __or__(self, other: _c_casted_longlong) -> "c_longlong": ...
    def __or__(self, other: clike_int) -> "c_longlong | c_ulonglong":
        return self._arith(other, int.__or__)


    @overload
    def __xor__(self, other: "c_ulonglong") -> "c_ulonglong": ...
    @overload
    def __xor__(self, other: _c_casted_longlong) -> "c_longlong": ...
    def __xor__(self, other: clike_int) -> "c_longlong | c_ulonglong":
        return self._arith(other, int.__xor__)


    def __neg__(self) -> "c_longlong":
        return c_longlong(-self.value)
    def __pos__(self) -> "c_longlong":
        return c_longlong(+self.value)
    def __invert__(self) -> "c_longlong":
        return c_longlong(~self.value)
    def __str__(self) -> str:
        return str(self.value)
    def __repr__(self) -> str:
        return str(self.value)

class c_ulonglong(c_integer, clike_longlong):

    @classmethod
    def cast(cls, x: int) -> int:
        """C++ unsigned long long wrapping"""
        return x % cls.UINT64_MAX

    def _arith(self, other: clike_int, op: Callable[[int, int], int]) -> "c_ulonglong":
        return c_ulonglong(op(self.value, int(other)))

    def __init__(self, value: int): self.value = self.cast(value)
    def __int__(self) -> int: return self.value
    
    def __add__(self, other: clike_int) -> "c_ulonglong":
        return self._arith(other, int.__add__)
    def __sub__(self, other: clike_int) -> "c_ulonglong":
        return self._arith(other, int.__sub__)
    def __mul__(self, other: clike_int) -> "c_ulonglong":
        return self._arith(other, int.__mul__)
    def __floordiv__(self, other: clike_int) -> "c_ulonglong":
        return self._arith(other, int.__floordiv__)
    def __truediv__(self, other: clike_int) -> "c_ulonglong":
        return self._arith(other, int.__floordiv__)
    def __mod__(self, other: clike_int) -> "c_ulonglong":
        return self._arith(other, int.__mod__)
    def __pow__(self, other: clike_int) -> "c_ulonglong":
        return self._arith(other, int.__pow__)
    
    def __lshift__(self, other: clike_int) -> "c_ulonglong":
        return c_ulonglong(self.value << int(other))
    def __rshift__(self, other: clike_int) -> "c_ulonglong":
        return c_ulonglong(self.value >> int(other))
    
    def __and__(self, other: clike_int) -> "c_ulonglong":
        return self._arith(other, int.__and__)
    def __or__(self, other: clike_int) -> "c_ulonglong":
        return self._arith(other, int.__or__)
    def __xor__(self, other: clike_int) -> "c_ulonglong":
        return self._arith(other, int.__xor__)
    def __neg__(self) -> "c_ulonglong":
        return c_ulonglong(-self.value)
    def __pos__(self) -> "c_ulonglong":
        return c_ulonglong(+self.value)
    def __invert__(self) -> "c_ulonglong":
        return c_ulonglong(~self.value)
    def __str__(self) -> str:
        return str(self.value)
    def __repr__(self) -> str:
        return str(self.value)
    
class c_float(Protocol):
    def __float__(self) -> float: ...
    def __add__(self, other: "c_float") -> "c_float": ...
    def __sub__(self, other: "c_float") -> "c_float": ...
    def __mul__(self, other: "c_float") -> "c_float": ...
    def __truediv__(self, other: "c_float") -> "c_float": ...
    def __pow__(self, other: "c_float") -> "c_float": ...
    def __neg__(self) -> "c_float": ...
    def __pos__(self) -> "c_float": ...

class c_double(Protocol):
    def __float__(self) -> float: ...
    def __add__(self, other: "c_double") -> "c_double": ...
    def __sub__(self, other: "c_double") -> "c_double": ...
    def __mul__(self, other: "c_double") -> "c_double": ...
    def __truediv__(self, other: "c_double") -> "c_double": ...
    def __pow__(self, other: "c_double") -> "c_double": ...
    def __neg__(self) -> "c_double": ...
    def __pos__(self) -> "c_double": ...

class c_char(Protocol):
    def __bytes__(self) -> bytes: ...
    def __len__(self) -> int: ...
    def decode(self, encoding: str = "utf-8") -> str: ...

class c_bool(Protocol):
    def __bool__(self) -> bool: ...
    def __int__(self) -> int: ...
    def __and__(self, other: "c_bool") -> "c_bool": ...
    def __or__(self, other: "c_bool") -> "c_bool": ...
    def __xor__(self, other: "c_bool") -> "c_bool": ...
    def __invert__(self) -> "c_bool": ...

__all__ = [
    "c_cast", "c_static_cast", "Literal", "Final",
    
    "c_void", "void",
    
    "c_exitcode",
    
    "c_global",
    
    "c_struct",
    "c_array",
    
    "c_bool",
    
    "c_char",
    "c_int",      "c_uint",
    "c_short",    "c_ushort",
    "c_long",     "c_ulong",
    "c_longlong", "c_ulonglong",
    "c_float",    "c_double"
]