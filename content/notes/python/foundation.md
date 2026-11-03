---
title: "Foundation"
date: "2026-10-27"
---

## 1\. Core Syntax & Data Types

Python is an interpreted, dynamically-typed, garbage-collected language. Indentation (4 spaces) defines scope blocks instead of braces `{}`.

### Variables, Dynamic Typing & Type Annotations

Python

```python
# Dynamic assignment
x = 42              # int
pi = 3.14159        # float
name = "Gemini"     # str
is_valid = True     # bool
empty = None        # NoneType

# Type annotations (Python 3.5+, enforced via static checkers like mypy)
user_id: int = 101
scores: list[float] = [98.5, 88.0, 91.2]
lookup: dict[str, int] = {"apples": 5, "oranges": 10}
```

### Primitive Operators

Python

```python
# Arithmetic
add = 10 + 3        # 13
sub = 10 - 3        # 7
mul = 10 * 3        # 30
div = 10 / 3        # 3.3333333333333335 (float division)
floor_div = 10 // 3 # 3 (integer/floor division)
mod = 10 % 3        # 1
power = 10 ** 3     # 1000

# Bitwise
bitwise_and = 6 & 3 # 0b110 & 0b011 = 0b010 (2)
bitwise_or  = 6 | 3 # 0b110 | 0b011 = 0b111 (7)
bitwise_xor = 6 ^ 3 # 5
left_shift  = 1 << 4# 16
right_shift = 16 >> 2# 4

# Logical & Identity
# and, or, not
# is (object identity), is not
# in (membership test), not in
```

### Built-in Collections

| Type | Syntax | Mutable? | Ordered? | Duplicate Keys/Values? |
| --- | --- | --- | --- | --- |
| `list` | `[1, 2, 3]` | Yes | Yes (insertion order) | Allowed |
| `tuple` | `(1, 2, 3)` | No | Yes (insertion order) | Allowed |
| `set` | `{1, 2, 3}` | Yes | No | Unique items only |
| `frozenset` | `frozenset([1, 2])` | No | No | Unique items only |
| `dict` | `{"k": "v"}` | Yes | Yes (Python 3.7+) | Unique keys; duplicate values |

Python

```python
# Slicing syntax: sequence[start:stop:step]
nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
sub = nums[2:7]      # [2, 3, 4, 5, 6]
stride = nums[::2]   # [0, 2, 4, 6, 8]
rev = nums[::-1]     # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
```

## 2\. Control Flow & Structural Pattern Matching

### Conditional Branching & Loops

Python

```python
# Standard branching
if x > 10:
    status = "high"
elif x == 10:
    status = "exact"
else:
    status = "low"

# Ternary expression
label = "even" if x % 2 == 0 else "odd"

# While loop with break, continue, else
count = 0
while count < 3:
    count += 1
else:
    # Executes only if loop completes without hitting a 'break'
    pass

# For loop with enumerate and zip
items = ["a", "b", "c"]
weights = [10, 20, 30]

for idx, (char, weight) in enumerate(zip(items, weights)):
    pass
```

### Structural Pattern Matching (`match-case`, Python 3.10+)

Python

```python
def parse_command(command: tuple | dict | str):
    match command:
        case "quit":
            return {"action": "exit"}
        case ("move", int(x), int(y)):
            return {"action": "translate", "coords": (x, y)}
        case ("move", *rest) if len(rest) > 2:
            return {"action": "multi_axis_move", "axes": rest}
        case {"type": "draw", "shape": "circle", "radius": r}:
            return {"action": "render", "radius": r}
        case _:
            return {"action": "unknown"}
```

## 3\. Functions, Closures & Scoping

### Signature Variations & Unpacking

Python

```python
# *args (variable positional), **kwargs (variable keyword)
# Positional-only parameters before \`/\`
# Keyword-only parameters after \`*\`
def configure_engine(prefix: str, /, timeout: int = 30, *, retries: int = 3, **kwargs):
    return f"{prefix}:{timeout}:{retries}:{kwargs}"

configure_engine("db", 45, retries=5, debug=True)

# Unpacking operators: * and **
args_list = [60]
kwargs_dict = {"retries": 10, "ssl": True}
configure_engine("api", *args_list, **kwargs_dict)
```

### Closures & Variable Scope (`global`, `nonlocal`)

Python

```python
def counter(initial: int = 0):
    current = initial
    
    def increment(step: int = 1) -> int:
        nonlocal current  # Binds to the nearest enclosing non-global scope
        current += step
        return current
        
    return increment

c = counter(10)
assert c() == 11
assert c(4) == 15
```

## 4\. Object-Oriented Programming (OOP) & Dunder Protocols

### Class Hierarchy, Inheritance & `super()`

Python

```python
class BaseService:
    def __init__(self, endpoint: str):
        self._endpoint = endpoint  # Protected convention

    def connect(self) -> None:
        pass

class SecureService(BaseService):
    def __init__(self, endpoint: str, token: str):
        super().__init__(endpoint)
        self.__token = token       # Name mangling: becomes _SecureService__token

    def connect(self) -> None:
        super().connect()
```

### Comprehensive Dunder/Data Model Protocol

Python

```python
class Vector:
    __slots__ = ("_x", "_y")  # Suppresses __dict__; optimizes memory allocation

    def __init__(self, x: float, y: float):
        self._x = x
        self._y = y

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    # String representation
    def __repr__(self) -> str:
        return f"Vector({self._x!r}, {self._y!r})"

    def __str__(self) -> str:
        return f"({self._x}, {self._y})"

    # Arithmetic operators
    def __add__(self, other: "Vector") -> "Vector":
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self._x + other._x, self._y + other._y)

    # Sequence / Container emulation
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> float:
        match index:
            case 0: return self._x
            case 1: return self._y
            case _: raise IndexError("Vector index out of range")

    # Equality & Hashing
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return self._x == other._x and self._y == other._y

    def __hash__(self) -> int:
        return hash((self._x, self._y))
```

## 5\. Functional Patterns & Comprehensions

Python

```python
# List, Dictionary, Set, and Generator Comprehensions
squares = [x**2 for x in range(10) if x % 2 == 0]
lookup_map = {f"k_{x}": x**2 for x in range(5)}
unique_mods = {x % 3 for x in [10, 11, 12, 13, 14]}
stream = (x**3 for x in range(100_000))  # Generator: zero upfront allocation

# Functional utilities: map, filter, lambdas
is_even = lambda x: x % 2 == 0
filtered = list(filter(is_even, [1, 2, 3, 4]))
mapped = list(map(lambda x: x * 10, filtered))

# Walrus Operator (:=) for in-place assignment (Python 3.8+)
line = "  batch_data_entry  "
if (cleaned := line.strip()).startswith("batch"):
    _ = cleaned.upper()
```

## 6\. Iterators, Generators & Context Managers

### Iteration Protocol & Coroutine Generators

Python

```python
class CountDown:
    def __init__(self, start: int):
        self.current = start

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

# Generator function with yield, send(), and yield from
def bidirectional_stream():
    val = yield "READY"
    while True:
        val = yield f"Received: {val}"

def delegating_pipeline():
    yield from range(3)          # Sub-generator delegation
    yield from ("a", "b", "c")
```

### Context Managers (`with` statement)

Python

```python
from contextlib import contextmanager

# Class-based Context Manager
class ManagedResource:
    def __enter__(self):
        # Setup resource
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup resource; return True to suppress exceptions
        return False

# Generator-based Context Manager
@contextmanager
def scoped_session():
    # Setup
    try:
        yield "Active Session"
    finally:
        # Tear down / rollback
        pass
```

## 7\. Metaprogramming: Decorators, Descriptors & Metaclasses

### Parameterized Decorators & Introspection

Python

```python
import functools
import time

def retry(times: int, backoff: float = 0.5):
    def decorator(func):
        @functools.wraps(func)  # Preserves __name__, __doc__, and type hints
        def wrapper(*args, **kwargs):
            last_err = None
            for _ in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as err:
                    last_err = err
                    time.sleep(backoff)
            raise last_err
        return wrapper
    return decorator
```

### Descriptors (`__get__`, `__set__`, `__delete__`)

Python

```python
class PositiveFloat:
    def __set_name__(self, owner, name):
        self.private_name = f"_{name}"

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return getattr(instance, self.private_name, 0.0)

    def __set__(self, instance, value: float):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{self.private_name} must be a positive number")
        setattr(instance, self.private_name, float(value))

class Product:
    price = PositiveFloat()

    def __init__(self, price: float):
        self.price = price
```

### Metaclasses

Metaclasses control class generation before instantiation.

Python

```python
class EnforceMethodsMeta(type):
    def __new__(mcs, name, bases, namespace):
        # Intercept class creation; validate or mutate attributes
        if name != "BasePlugin" and "execute" not in namespace:
            raise TypeError(f"Class {name} must define an 'execute' method.")
        return super().__new__(mcs, name, bases, namespace)

class BasePlugin(metaclass=EnforceMethodsMeta):
    pass
```

## 8\. Asynchronous Programming (`asyncio`)

Python executes asynchronous concurrency using an underlying cooperative event loop.

Python

```python
import asyncio

async def fetch_record(record_id: int) -> dict:
    await asyncio.sleep(0.05)  # Yields control back to the event loop
    return {"id": record_id, "data": "payload"}

async def main():
    # Concurrent execution with TaskGroups (Python 3.11+)
    async with asyncio.TaskGroup() as tg:
        t1 = tg.create_task(fetch_record(1))
        t2 = tg.create_task(fetch_record(2))
    
    # Both tasks complete when the TaskGroup context exits
    results = [t1.result(), t2.result()]
    return results

# Invocation:
# asyncio.run(main())
```

## 9\. Modern Memory Management & Concurrency Primitives

- **Memory Management:** CPython relies on reference counting complemented by a generational cyclic garbage collector (Generation 0, 1, 2) to eliminate circular references.
- **Global Interpreter Lock (GIL):** A mutex preventing multiple native threads from executing Python bytecode simultaneously. Multi-threading is optimal for I/O-bound tasks (`threading`), while CPU-bound tasks rely on distinct subprocesses (`multiprocessing` or `concurrent.futures.ProcessPoolExecutor`) to bypass the GIL.
