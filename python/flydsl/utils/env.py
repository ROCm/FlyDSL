import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

T = TypeVar("T")


class EnvOption(Generic[T]):
    def __init__(
        self,
        default: T,
        env_var: Optional[str] = None,
        description: str = "",
        validator: Optional[Callable[[T], bool]] = None,
    ):
        self.default = default
        self.env_var = env_var
        self.description = description
        self.validator = validator
        self.name: Optional[str] = None

    def __set_name__(self, owner: type, name: str):
        self.name = name

    def parse_value(self, raw: str) -> T:
        raise NotImplementedError

    def __get__(self, obj: Optional[object], objtype: Optional[type] = None) -> T:
        if obj is None:
            return self  # type: ignore

        if self.env_var is None:
            raise RuntimeError(
                f"EnvOption '{self.name or '<unknown>'}' has no env_var set. "
                "EnvOption must be used as a class attribute in an EnvManager subclass."
            )

        raw = os.environ.get(self.env_var)
        if raw is None:
            return self.default

        try:
            value = self.parse_value(raw)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to parse environment variable {self.env_var}={raw!r}: {e}") from e

        if self.validator is not None and not self.validator(value):
            raise ValueError(f"Invalid value for environment variable {self.env_var}: {value!r}")

        return value


class OptBool(EnvOption[bool]):
    def __init__(
        self,
        default: bool = False,
        env_var: Optional[str] = None,
        description: str = "",
    ):
        super().__init__(default, env_var, description)

    def parse_value(self, raw: str) -> bool:
        return raw.lower() in ("1", "true", "yes", "on")


class OptInt(EnvOption[int]):
    def __init__(
        self,
        default: int = 0,
        env_var: Optional[str] = None,
        description: str = "",
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ):
        validator = None
        if min_value is not None or max_value is not None:

            def validator(v: int) -> bool:
                if min_value is not None and v < min_value:
                    return False
                if max_value is not None and v > max_value:
                    return False
                return True

        super().__init__(default, env_var, description, validator)
        self.min_value = min_value
        self.max_value = max_value

    def parse_value(self, raw: str) -> int:
        return int(raw)


class OptStr(EnvOption[str]):
    def __init__(
        self,
        default: str = "",
        env_var: Optional[str] = None,
        description: str = "",
        choices: Optional[list[str]] = None,
    ):
        validator = None
        if choices is not None:

            def validator(v: str) -> bool:
                return v in choices

        super().__init__(default, env_var, description, validator)
        self.choices = choices

    def parse_value(self, raw: str) -> str:
        return raw


E = TypeVar("E", int, str)


class OptList(EnvOption[list[E]]):
    def __init__(
        self,
        default: Optional[list[E]] = None,
        env_var: Optional[str] = None,
        description: str = "",
        separator: str = ",",
        element_type: type = str,
    ):
        super().__init__(default or [], env_var, description)
        self.separator = separator
        self.element_type = element_type

    def parse_value(self, raw: str) -> list[E]:
        if not raw:
            return []
        items = [s.strip() for s in raw.split(self.separator)]
        if self.element_type is int:
            return [int(s) for s in items]
        return items


class EnvManagerMeta(type):
    def __new__(mcs, name: str, bases: tuple, namespace: dict, **kwargs):
        parent_prefix = None
        env_bases = [b for b in bases if hasattr(b, "env_prefix")]
        if len(env_bases) > 1:
            raise TypeError(f"EnvManager subclass '{name}' can only inherit from one EnvManager parent")

        parent_prefix = env_bases[0].env_prefix if env_bases else None

        if "env_prefix" in namespace:
            child_prefix = namespace["env_prefix"]
            if parent_prefix:
                namespace["env_prefix"] = f"{parent_prefix}_{child_prefix}"
        elif parent_prefix:
            namespace["env_prefix"] = parent_prefix

        cls = super().__new__(mcs, name, bases, namespace)

        options: Dict[str, EnvOption] = {}
        for key, value in namespace.items():
            if isinstance(value, EnvOption):
                if value.env_var is None:
                    upper_key = re.sub(r"([a-z])([A-Z])", r"\1_\2", key).upper()
                    value.env_var = f"{cls.env_prefix}_{upper_key}"
                options[key] = value

        cls.options = options
        return cls


class EnvManager(metaclass=EnvManagerMeta):
    env_prefix: str = "FLYDSL"
    options: Dict[str, EnvOption]

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.options}

    @classmethod
    def help(cls) -> str:
        lines = [f"{cls.__name__} Options:", ""]
        for name, opt in cls.options.items():
            desc = opt.description or "No description"
            lines.append(f"  {name}:")
            lines.append(f"    Environment: {opt.env_var}")
            lines.append(f"    Default: {opt.default!r}")
            lines.append(f"    Description: {desc}")
            lines.append("")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.help()


class CompileEnvManager(EnvManager):
    env_prefix = "COMPILE"

    opt_level = OptInt(2, min_value=0, max_value=3, description="Optimization level")


class DebugEnvManager(EnvManager):
    env_prefix = "DEBUG"

    dump_asm = OptBool(False, description="Dump ASM to file")
    dump_ir = OptBool(False, description="Dump IR to file")
    dump_dir = OptStr(str(Path.home() / ".flydsl" / "debug"), description="Directory for dumping IR")

    # Logging options
    log_level = OptStr("WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"], description="Logging level")
    log_to_file = OptStr("", description="Log file path, empty to disable file logging")
    log_to_console = OptBool(False, description="Enable console logging")

    # MLIR pass manager options
    print_after_all = OptBool(False, description="Print IR after each MLIR pass")
    enable_debug_info = OptBool(True, description="Generate debug info in compiled code")
    enable_verifier = OptBool(True, description="Verify IR module")


class RuntimeEnvManager(EnvManager):
    env_prefix = "RUNTIME"

    cache_dir = OptStr(str(Path.home() / ".flydsl" / "cache"), description="Directory for caching compiled kernels")
    enable_cache = OptBool(True, description="Enable kernel caching")


compile = CompileEnvManager()
debug = DebugEnvManager()
runtime = RuntimeEnvManager()

__all__ = [
    "compile",
    "debug",
    "runtime",
]
