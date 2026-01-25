import ctypes
import numpy as np
import operator
from typing_extensions import deprecated
from functools import reduce
from typing import (
    Generic,
    Protocol,
    Union,
    Any,
    List,
    Type,
    TypeVar,
    overload,
    runtime_checkable,
    get_origin,
)
from types import FunctionType
from dataclasses import dataclass
from abc import ABC, abstractmethod


class NumericType:
    pass


class Int32:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Int32({self.value})"
