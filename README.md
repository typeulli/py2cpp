# py2cpp

A Python-to-C++ translator that converts a restricted subset of Python into clean and readable C++ code. It offers both Ahead-of-Time (AOT) compilation through a command-line tool and Just-in-Time (JIT) compilation via a decorator.

[Online Simulator](https://typeulli.com/py2cpp)

## Key Features

- **Dual Compilation Modes**:
    - **Ahead-of-Time (AOT)**: Convert entire Python files to C++ source code or executables using the `py2cpp` command-line tool.
    - **Just-in-Time (JIT)**: Use the `@jit` decorator to compile and run specific Python functions in C++ at runtime for a performance boost.
- **Readable C++ Code**: Translates Python syntax into human-readable and clean C++ code.
- **Type-Hint Driven**: Utilizes Python type hints to generate corresponding C++ types.
- **C-Style Data Structures**: Define C-like structures directly in Python using the `@c_struct` decorator.
- **Fine-Grained Type Control**: Provides special types like `c_int`, `c_double`, `c_char`, etc., for precise control over the generated C++ data types.
- **Python Interoperability**: Allows passing Python objects to and from JIT-compiled functions.

**Note:**

> The actual translation may vary based on the specific implementation details of py2cpp.
>
> This project doesn't optimize your code for performance or memory usage.

## Project Goals

1. Provide a static translation layer. (Without using A.I.)
2. Offer a simplified Python-like syntax that maps cleanly to C++ constructs.
3. Covering all python syntax constructs as much as possible.
4. Minimize boilerplate on both the Python input side and the C++ output side.

## Installation

```bash
pip install py2cpp
```
## Usage

### Command-Line Usage (AOT)

The `py2cpp` command-line tool translates `.py` files into `.cpp` files or executables.

```bash
# Translate to C++
py2cpp input.py

# Translate to a C++ executable
py2cpp input.py -c exe

# Specify output file
py2cpp input.py -o output.cpp
py2cpp input.py -o output.exe
```

### JIT-Compilation Usage

Use the `@jit` decorator for on-the-fly compilation and execution of Python functions.

```python
from py2cpp.core.jit import jit

@jit
def add(a: int, b: int) -> int:
    return a + b

# The 'add' function is now a compiled C++ function
result = add(5, 10)
print(result)
```

## Examples

### AOT Compilation Example

This Python code:
```python
def add(a: int, b: int) -> int:
    return a + b

x = input()
y = 5

print(add(int(x), y))
```
translates to this C++ code:
```cpp  
#include <iostream>
#include <string>

long add(long a, long b) {
    return a + b;
}

int main() {
    std::string x;
    std::getline(std::cin, x);
    long y = 5;
    std::cout << add(std::stol(x), y) << std::endl;
    return 0;
}
```

### JIT Compilation Example

Here is a simple example of using the `@jit` decorator to accelerate a function:

```python
from py2cpp.core.jit import jit
from time import time

@jit
def fast_add(a: int, b: int) -> int:
    n = a
    for _ in range(100000000):
        n += b
    return n

start = time()
result = fast_add(10, 20)
print(f"Result: {result}")
print(f"Execution time: {time() - start:.4f}s")
```

## License
MIT License. See `LICENSE` file for details.