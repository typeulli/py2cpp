# py2cpp

A Python-to-C++ translator that converts a restricted subset of Python into clean and readable C++ code.

[Online Simulator](https:://typeulli.com/py2cpp)

## Key Features

- Deterministic translation from Python syntax into valid C++.
- Predictable type mapping and simplified semantics.
- Modular architecture for extending grammar and translation rules.
- CLI interface for converting `.py` source files into `.cpp`.

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
git clone https://github.com/typeulli/py2cpp.git
cd py2cpp
pip install -r requirements.txt
```
## Usage
```bash
py2cpp input.py
py2cpp input.py -c exe
py2cpp input.py -o output.cpp
py2cpp input.py -o output.exe
```

## Example
```python
def add(a: int, b: int) -> int:
    return a + b

x = input()
y = 5

print(add(int(x), y))
```
translates to
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

## License
MIT License. See `LICENSE` file for details.