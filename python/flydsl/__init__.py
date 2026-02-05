"""
FlyDSL Python Package

This package provides Python bindings for the Fly dialect (MLIR DSL for GPU kernels).

IMPORTANT: Since we use MLIR_BINDINGS_PYTHON_NB_DOMAIN="flydsl", this package has its own
separate instances of Context, Type, etc. that are NOT compatible with upstream mlir.ir objects.

Usage:
    import flydsl
    
    # Use flydsl's own Context (not mlir.ir.Context!)
    ctx = flydsl.Context()
    flydsl.register_dialect(ctx)
    
    # Now use Fly types
    from flydsl._mlir._mlir_libs._fly import IntTupleType
    t = IntTupleType.get(42, ctx)
"""

# Lazy imports for core types to avoid circular dependencies
_CORE_TYPE_NAMES = ['Context', 'Module', 'Location', 'InsertionPoint', 'Type', 'Attribute', 'Value']
_core_types_cache = {}

def __getattr__(name):
    """Lazy load core types from _fly extension"""
    if name in _CORE_TYPE_NAMES:
        if name not in _core_types_cache:
            from ._mlir._mlir_libs._fly import (
                Context, Module, Location, InsertionPoint, Type, Attribute, Value
            )
            _core_types_cache.update({
                'Context': Context,
                'Module': Module,
                'Location': Location,
                'InsertionPoint': InsertionPoint,
                'Type': Type,
                'Attribute': Attribute,
                'Value': Value,
            })
        return _core_types_cache[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Also keep upstream MLIR available for compatibility (but mark it as separate)
try:
    from mlir import ir as _upstream_mlir_ir
except ImportError:
    _upstream_mlir_ir = None  # Optional dependency


class _LazyModuleLoader:
    """
    Lazy loader for Fly extension modules.
    
    Delays importing _fly/_fly_rocdl until first access to avoid
    circular imports and reduce startup time.
    """
    def __init__(self, module_path, module_name):
        self._module_path = module_path
        self._module_name = module_name
        self._module = None
        self._load_error = None
    
    def _load(self):
        """Load the module on first access, cache result or error."""
        if self._module is not None:
            return self._module
        
        if self._load_error is not None:
            raise self._load_error
        
        try:
            if self._module_path == '_fly':
                from ._mlir._mlir_libs import _fly
                self._module = _fly
            elif self._module_path == '_fly_rocdl':
                from ._mlir._mlir_libs import _fly_rocdl
                self._module = _fly_rocdl
            else:
                raise ImportError(f"Unknown module path: {self._module_path}")
        except Exception as e:
            self._load_error = ImportError(f"Failed to load {self._module_name}: {e}")
            raise self._load_error
        
        return self._module
    
    def __getattr__(self, name):
        """Proxy attribute access to the actual module."""
        return getattr(self._load(), name)
    
    def __dir__(self):
        """Support tab completion in interactive sessions."""
        try:
            return dir(self._load())
        except Exception:
            return []


# Lazy-loaded module proxies
_fly = _LazyModuleLoader('_fly', 'Fly extension')
_fly_rocdl = _LazyModuleLoader('_fly_rocdl', 'FlyROCDL extension')


def _check_extension_loaded(loader):
    """Check if extension is successfully loaded."""
    try:
        loader._load()
        return True
    except Exception:
        return False


# Lazy extension status checks
@property
def _FLY_EXTENSION_LOADED():
    return _check_extension_loaded(_fly)

@property  
def _FLY_ROCDL_EXTENSION_LOADED():
    return _check_extension_loaded(_fly_rocdl)


def initialize():
    """
    Initialize FlyDSL environment.
    
    This function will:
    1. Create an MLIR context in the flydsl domain
    2. Preload _fly and _fly_rocdl extension modules
    
    Returns:
        Context: flydsl.Context instance (NOT mlir.ir.Context!)
    
    Example:
        import flydsl
        ctx = flydsl.initialize()
        flydsl.register_dialect(ctx)
        
        # Now use Fly types
        from flydsl._mlir._mlir_libs._fly import IntTupleType
        t = IntTupleType.get(42, ctx)
    """
    # Context is loaded lazily via __getattr__
    ctx = __getattr__('Context')()
    
    # Trigger lazy loading early to detect issues
    try:
        _fly._load()
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to preload _fly extension: {e}")
    
    try:
        _fly_rocdl._load()
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to preload _fly_rocdl extension: {e}")
    
    return ctx


def register_dialect(context=None):
    """
    Register Fly dialect with a flydsl Context.
    
    Args:
        context: flydsl.Context (NOT mlir.ir.Context!). Optional - if None, creates a new one.
    
    Returns:
        The context with Fly dialect registered
    
    Example:
        import flydsl
        
        ctx = flydsl.Context()  # Use flydsl's Context!
        flydsl.register_dialect(ctx)
    """
    # Load _fly module if not already loaded
    try:
        _fly._load()
    except Exception as e:
        raise RuntimeError(f"Fly extension not loaded: {e}")
    
    # Create a flydsl Context if none provided
    if context is None:
        ContextClass = __getattr__('Context')
        context = ContextClass()
    
    # Type check: ensure it's a flydsl Context, not upstream mlir.ir.Context
    ContextClass = __getattr__('Context')
    if not isinstance(context, ContextClass):
        raise TypeError(
            f"Expected flydsl.Context, got {type(context)}. "
            f"Note: flydsl.Context is NOT compatible with mlir.ir.Context "
            f"due to MLIR_BINDINGS_PYTHON_NB_DOMAIN='flydsl'."
        )
    
    # Call the C++ registration function
    _fly._register_dialect(context)
    
    return context


# Import compiler (optional, may fail if dependencies missing)
_COMPILER_AVAILABLE = False
try:
    from .compiler.compiler import compile
    _COMPILER_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Compiler not available: {e}")
    compile = None


__all__ = [
    'Context', 'Module', 'Location', 'InsertionPoint', 'Type', 'Attribute', 'Value',
    'initialize', 'register_dialect', '_fly', '_fly_rocdl', '_COMPILER_AVAILABLE',
    'compile',
]
