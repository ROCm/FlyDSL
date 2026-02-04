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
    延迟加载Fly扩展模块
    
    这个类实现了延迟导入机制，确保在第一次访问_fly或_fly_rocdl模块时：
    1. 已经有一个MLIR context存在
    2. 如果没有，自动创建一个默认context
    
    这样用户就不需要记住特定的导入顺序了。
    """
    def __init__(self, module_path, module_name):
        self._module_path = module_path
        self._module_name = module_name
        self._module = None
        self._load_attempted = False
        self._load_error = None
    
    def _ensure_context_exists(self):
        """确保Context类可用（不创建实例）"""
        # 注意：不再创建临时Context实例，避免内存泄漏
        # Context会在需要时由GlobalRAIIMLIRContext创建
        pass
    
    def _load(self):
        """延迟加载模块"""
        if self._module is not None:
            return self._module
        
        if self._load_attempted:
            if self._load_error:
                raise self._load_error
            return self._module
        
        self._load_attempted = True
        
        try:
            # 在导入前确保有context
            self._ensure_context_exists()
            
            # 动态导入模块
            if self._module_path == '_fly':
                from ._mlir._mlir_libs import _fly
                self._module = _fly
            elif self._module_path == '_fly_rocdl':
                from ._mlir._mlir_libs import _fly_rocdl
                self._module = _fly_rocdl
            else:
                raise ImportError(f"Unknown module path: {self._module_path}")
            
        except Exception as e:
            self._load_error = ImportError(
                f"Failed to load {self._module_name}: {e}"
            )
            raise self._load_error
        
        return self._module
    
    def __getattr__(self, name):
        """代理所有属性访问到实际的模块"""
        module = self._load()
        return getattr(module, name)
    
    def __dir__(self):
        """支持tab补全"""
        try:
            module = self._load()
            return dir(module)
        except Exception:
            return []


# 创建延迟加载的模块代理
_fly = _LazyModuleLoader('_fly', 'Fly extension')
_fly_rocdl = _LazyModuleLoader('_fly_rocdl', 'FlyROCDL extension')


def _check_extension_loaded(loader):
    """检查扩展是否成功加载"""
    try:
        loader._load()
        return True
    except Exception:
        return False


# 延迟检查扩展状态
@property
def _FLY_EXTENSION_LOADED():
    return _check_extension_loaded(_fly)

@property  
def _FLY_ROCDL_EXTENSION_LOADED():
    return _check_extension_loaded(_fly_rocdl)


def initialize():
    """
    初始化FlyDSL环境
    
    这个函数会:
    1. 创建一个flydsl domain的MLIR context
    2. 预加载_fly和_fly_rocdl扩展模块
    
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
    # Context通过__getattr__自动加载
    ctx = __getattr__('Context')()
    
    # 触发延迟加载以尽早发现问题
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
    Register Fly dialect with a flydsl Context
    
    Args:
        context: flydsl.Context (NOT mlir.ir.Context!). Optional - if None, creates a new one.
    
    Returns:
        The context with Fly dialect registered
    
    Example:
        import flydsl
        
        ctx = flydsl.Context()  # Use flydsl's Context!
        flydsl.register_dialect(ctx)
    """
    # 触发_fly模块加载（如果还未加载）
    try:
        _fly._load()
    except Exception as e:
        raise RuntimeError(f"Fly extension not loaded: {e}")
    
    # 如果没有传入context，创建一个flydsl的Context
    if context is None:
        ContextClass = __getattr__('Context')
        context = ContextClass()
    
    # 类型检查：确保是flydsl的Context，不是上游mlir.ir.Context
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
