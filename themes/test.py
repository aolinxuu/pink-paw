#!/usr/bin/env python3
# This is a comment to test italic green comment styling
# Comments should appear in #4c835b with italic font
"""
Multi-line comment block
Testing comment highlighting across multiple lines
Should also be italic and green
"""

import sys
import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter, namedtuple
import json
import re
import math

# ============================================================================
# CONSTANTS AND VARIABLES SECTION
# Testing: constant.numeric, constant.language, variable
# ============================================================================

# Numbers (constant.numeric) - should be #F78C6C
integer_value = 42
float_value = 3.14159
hex_value = 0xFF00AB
binary_value = 0b101010
octal_value = 0o755
scientific = 1.5e-10

# Constants (constant.language) - should be #F78C6C
BOOLEAN_TRUE = True
BOOLEAN_FALSE = False
NULL_VALUE = None

# Variables (variable) - should be #ffffff (white)
user_name = "Alice"
user_age = 30
is_active = True
empty_list = []

# ============================================================================
# STRING TESTING
# Testing: string scope - should be #C3E88D
# ============================================================================

single_quote_string = 'This is a single quote string'
double_quote_string = "This is a double quote string"
multiline_string = """
This is a multiline string
that spans multiple lines
for testing purposes
"""

# String interpolation
name = "Bob"
formatted_string = f"Hello, {name}! You are {user_age} years old."
percent_format = "Value: %s, Number: %d" % ("test", 100)
format_method = "The answer is {}".format(42)

# Escape characters (constant.character.escape) - should be #89DDFF
escaped_string = "Line 1\nLine 2\tTabbed\r\nWindows newline"
unicode_string = "\u0041\u0042\u0043"

# Regular expressions (string.regexp) - should be #89DDFF
pattern = r'\d+\.\d+'
email_pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
url_pattern = r'https?://(?:www\.)?[\w\-\.]+\.[\w]{2,}'

# ============================================================================
# KEYWORD AND STORAGE TESTING
# Testing: keyword, storage.type, storage.modifier - should be #ea92d3
# ============================================================================

# Control flow keywords
def test_control_flow():
    if True:
        pass
    elif False:
        pass
    else:
        pass
    
    for i in range(10):
        if i == 5:
            break
        elif i == 3:
            continue
        else:
            pass
    
    while True:
        break
    
    try:
        raise Exception("test")
    except Exception as e:
        pass
    finally:
        pass
    
    with open("file.txt") as f:
        data = f.read()
    
    return None

# Lambda (keyword) - should be #ea92d3
square = lambda x: x ** 2
add = lambda a, b: a + b

# Keywords: def, class, return, yield, await, async
async def async_function():
    await some_coroutine()
    yield 42
    return True

# Global, nonlocal keywords
global_var = 100

def outer():
    outer_var = 50
    const variable = 50
    
    def inner():
        nonlocal outer_var
        global global_var
        outer_var += 1
        global_var += 1

# ============================================================================
# OPERATOR AND PUNCTUATION TESTING
# Testing: keyword.control, punctuation - should be #89DDFF
# ============================================================================

# Arithmetic operators
result = 10 + 5 - 3 * 2 / 4 % 3 ** 2 // 2

# Comparison operators
comparison = (5 > 3) and (2 < 4) or (1 == 1) and (2 != 3)
identity = x is None
membership = item in collection

# Bitwise operators
bitwise = 5 & 3 | 2 ^ 1 << 2 >> 1 ~ 4

# Assignment operators
x += 1
y -= 2
z *= 3
a /= 4
b //= 5
c %= 6
d **= 7

# Punctuation: brackets, braces, parentheses, commas, colons, semicolons
list_example = [1, 2, 3, 4, 5]
dict_example = {"key": "value", "number": 42}
tuple_example = (1, 2, 3)
set_example = {1, 2, 3, 4}

# ============================================================================
# FUNCTION DEFINITIONS AND CALLS
# Testing: entity.name.function, meta.function-call - should be #ffce6b
# ============================================================================

# Function definition
def simple_function(param1, param2="default"):
    """Docstring for the function"""
    return param1 + param2

def complex_function(a, b, *args, **kwargs):
    result = a + b
    for arg in args:
        result += arg
    return result

# Function calls
output = simple_function(10, 20)
result = complex_function(1, 2, 3, 4, 5, key="value")
builtin_call = print("Hello, World!")
len_call = len([1, 2, 3])
max_value = max([1, 5, 3, 9, 2])

# Method calls
text = "hello world"
upper_text = text.upper()
replaced = text.replace("world", "Python")

# ============================================================================
# CLASS DEFINITIONS
# Testing: entity.name (class names), support.type - should be #FFCB6B
# ============================================================================

class BaseClass:
    """Base class for testing"""
    
    class_variable = "shared"
    
    def __init__(self, name):
        self.name = name
        self.instance_var = []
    
    def instance_method(self):
        return f"Instance: {self.name}"
    
    @classmethod
    def class_method(cls):
        return cls.class_variable
    
    @staticmethod
    def static_method():
        return "Static method"
    
    @property
    def name_property(self):
        return self._name
    
    @name_property.setter
    def name_property(self, value):
        self._name = value

class DerivedClass(BaseClass):
    """Derived class testing inheritance"""
    
    def __init__(self, name, age):
        super().__init__(name)
        self.age = age
    
    def __str__(self):
        return f"{self.name}: {self.age}"
    
    def __repr__(self):
        return f"DerivedClass(name={self.name}, age={self.age})"

# Testing support.type and support.class
class GenericClass:
    pass

# ============================================================================
# DECORATORS
# Testing: decorators - should be #82AAFF with italic
# ============================================================================

def decorator_function(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

@decorator_function
def decorated_function():
    return "I'm decorated!"

@property
def my_property(self):
    return self._value

# Multiple decorators
@decorator_function
@staticmethod
def multi_decorated():
    pass

# ============================================================================
# LANGUAGE VARIABLES
# Testing: variable.language - should be #FF5370 with italic
# ============================================================================

class TestSelf:
    def __init__(self):
        self.value = 42  # 'self' should be #FF5370 italic
    
    def method(self):
        return self.value
    
    @classmethod
    def class_method(cls):  # 'cls' should be #FF5370 italic
        return cls.__name__

# ============================================================================
# BUILT-IN FUNCTIONS AND TYPES
# Testing: support.function, support.type - various colors
# ============================================================================

# Built-in functions
result = len([1, 2, 3])
text = str(42)
number = int("100")
floating = float("3.14")
listed = list((1, 2, 3))
dictionary = dict(a=1, b=2)
printed = print("output")
opened = open("file.txt", "r")
sorted_list = sorted([3, 1, 4, 1, 5])
mapped = map(lambda x: x * 2, [1, 2, 3])
filtered = filter(lambda x: x > 0, [-1, 0, 1, 2])
zipped = zip([1, 2], ['a', 'b'])
enumerated = enumerate(['a', 'b', 'c'])

# Type annotations (support.type) - should be #B2CCD6
def typed_function(name: str, age: int) -> bool:
    return True

def generic_function(items: list[int]) -> dict[str, int]:
    return {"count": len(items)}

# ============================================================================
# COMPREHENSIONS
# Testing various scopes in list/dict/set comprehensions
# ============================================================================

# List comprehension
squares = [x**2 for x in range(10)]
evens = [n for n in range(20) if n % 2 == 0]
matrix = [[i*j for j in range(5)] for i in range(5)]

# Dictionary comprehension
squared_dict = {k: k**2 for k in range(1, 11)}
filtered_dict = {k: v for k, v in {"a": 1, "b": 2}.items() if v > 1}

# Set comprehension
unique_squares = {x**2 for x in range(-5, 6)}

# Generator expression
gen = (x**2 for x in range(100))

# ============================================================================
# EXCEPTION HANDLING
# Testing exception-related keywords and types
# ============================================================================

try:
    risky_operation = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")
except (TypeError, ValueError) as e:
    print(f"Multiple exceptions: {e}")
except Exception as e:
    raise RuntimeError("Something went wrong") from e
else:
    print("No exceptions occurred")
finally:
    print("Cleanup code")

# Custom exception
class CustomError(Exception):
    """Custom exception class"""
    pass

# Raising exceptions
if error_condition:
    raise CustomError("This is a custom error")

# Assert statements
assert 1 + 1 == 2, "Math is broken!"
assert isinstance(value, int)

# ============================================================================
# DATA STRUCTURES AND COLLECTIONS
# Testing dictionary, list, set operations
# ============================================================================

# Dictionary operations
person = {
    "name": "Alice",
    "age": 30,
    "email": "alice@example.com",
    "tags": ["python", "developer"],
    "metadata": {
        "created": "2024-01-01",
        "updated": "2024-12-31"
    }
}

# Dictionary methods
keys = person.keys()
values = person.values()
items = person.items()
name = person.get("name", "Unknown")
person.update({"city": "New York"})
removed = person.pop("age", None)

# List operations
numbers = [1, 2, 3, 4, 5]
numbers.append(6)
numbers.extend([7, 8, 9])
numbers.insert(0, 0)
popped = numbers.pop()
numbers.remove(3)
numbers.reverse()
numbers.sort()
numbers.clear()

# Set operations
set_a = {1, 2, 3, 4, 5}
set_b = {4, 5, 6, 7, 8}
union = set_a | set_b
intersection = set_a & set_b
difference = set_a - set_b
symmetric_diff = set_a ^ set_b

# ============================================================================
# SPECIAL METHODS (MAGIC METHODS)
# Testing dunder methods
# ============================================================================

class MagicClass:
    def __init__(self, value):
        self.__value = value
    
    def __str__(self):
        return f"MagicClass({self.__value})"
    
    def __repr__(self):
        return f"MagicClass(value={self.__value})"
    
    def __len__(self):
        return len(str(self.__value))
    
    def __getitem__(self, key):
        return self.__value[key]
    
    def __setitem__(self, key, value):
        self.__value[key] = value
    
    def __delitem__(self, key):
        del self.__value[key]
    
    def __iter__(self):
        return iter(self.__value)
    
    def __next__(self):
        return next(self.__value)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def __call__(self, *args, **kwargs):
        return self.__value
    
    def __add__(self, other):
        return self.__value + other
    
    def __sub__(self, other):
        return self.__value - other
    
    def __mul__(self, other):
        return self.__value * other
    
    def __eq__(self, other):
        return self.__value == other
    
    def __lt__(self, other):
        return self.__value < other
    
    def __le__(self, other):
        return self.__value <= other

# ============================================================================
# IMPORTS AND MODULES
# Testing import statements
# ============================================================================

import os
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Union, Any
from collections.abc import Iterable, Iterator, Callable
import numpy as np
import pandas as pd
from datetime import datetime as dt

# Relative imports
from .module import function
from ..package import Class
from ...parent import utility

# Import with alias
import matplotlib.pyplot as plt
from collections import defaultdict as dd

# ============================================================================
# TYPE HINTS AND ANNOTATIONS
# Testing type annotation scopes
# ============================================================================

from typing import TypeVar, Generic, Protocol

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self.items: List[T] = []
    
    def push(self, item: T) -> None:
        self.items.append(item)
    
    def pop(self) -> T:
        return self.items.pop()
    
    def peek(self) -> Optional[T]:
        return self.items[-1] if self.items else None

def process_data(
    data: Dict[str, Any],
    filter_func: Callable[[Any], bool],
    limit: Optional[int] = None
) -> List[Tuple[str, Any]]:
    """Process data with filtering"""
    result: List[Tuple[str, Any]] = []
    for key, value in data.items():
        if filter_func(value):
            result.append((key, value))
    return result[:limit] if limit else result

# ============================================================================
# MAIN EXECUTION
# Testing if __name__ == "__main__" pattern
# ============================================================================

if __name__ == "__main__":
    print("Running theme test code...")
    
    # Create instances
    obj = BaseClass("Test")
    derived = DerivedClass("Alice", 25)
    
    # Call functions
    result = simple_function(10, 20)
    decorated = decorated_function()
    
    # Use data structures
    data = {"a": 1, "b": 2, "c": 3}
    filtered = {k: v for k, v in data.items() if v > 1}
    
    print("All syntax elements tested!")
    print(f"Result: {result}")
    print(f"Derived: {derived}")