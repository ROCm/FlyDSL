"""
FlyDSL Interoperability Module

Provides utilities for converting between flydsl domain types and upstream mlir types
using the Capsule (_CAPIPtr/_CAPICreate) mechanism.
"""

from mlir import ir as upstream_ir


def to_upstream(obj):
    """
    Convert a flydsl domain object to its upstream mlir equivalent.
    
    Uses the Capsule mechanism (_CAPIPtr/_CAPICreate) to share the underlying
    C pointer between domains.
    
    Args:
        obj: A flydsl domain object (Type, Attribute, Value, etc.)
        
    Returns:
        The equivalent upstream mlir object, or the original if conversion not possible
    """
    if obj is None:
        return None
    
    # If it's already an upstream type, return as-is
    if type(obj).__module__.startswith('mlir.'):
        return obj
    
    # Try Capsule conversion
    if hasattr(obj, '_CAPIPtr'):
        capsule = obj._CAPIPtr
        type_name = type(obj).__name__
        
        # First try exact class match
        upstream_cls = getattr(upstream_ir, type_name, None)
        if upstream_cls and hasattr(upstream_cls, '_CAPICreate'):
            try:
                return upstream_cls._CAPICreate(capsule)
            except Exception:
                pass
        
        # For Type subclasses (like CopyAtomUniversalCopyType), use base Type class
        # Check if this is a Type by looking at the capsule name
        capsule_name = str(capsule)
        if 'Type._CAPIPtr' in capsule_name:
            try:
                return upstream_ir.Type._CAPICreate(capsule)
            except Exception:
                pass
        elif 'Attribute._CAPIPtr' in capsule_name:
            try:
                return upstream_ir.Attribute._CAPICreate(capsule)
            except Exception:
                pass
        elif 'Value._CAPIPtr' in capsule_name:
            try:
                return upstream_ir.Value._CAPICreate(capsule)
            except Exception:
                pass
    
    return obj


def to_upstream_types(types):
    """
    Convert a sequence of flydsl types to upstream mlir types.
    
    Args:
        types: A sequence of Type objects
        
    Returns:
        A list of upstream mlir Type objects
    """
    return [to_upstream(t) for t in types]


def to_flydsl(obj, flydsl_ir):
    """
    Convert an upstream mlir object to its flydsl domain equivalent.
    
    Args:
        obj: An upstream mlir object
        flydsl_ir: The flydsl ir module (flydsl._mlir.ir)
        
    Returns:
        The equivalent flydsl domain object
    """
    if obj is None:
        return None
    
    # If it's already a flydsl type, return as-is
    if 'flydsl' in type(obj).__module__:
        return obj
    
    # Try Capsule conversion
    if hasattr(obj, '_CAPIPtr'):
        capsule = obj._CAPIPtr
        type_name = type(obj).__name__
        
        # Try to find the corresponding flydsl class
        flydsl_cls = getattr(flydsl_ir, type_name, None)
        if flydsl_cls and hasattr(flydsl_cls, '_CAPICreate'):
            try:
                return flydsl_cls._CAPICreate(capsule)
            except Exception:
                pass
    
    return obj


__all__ = ['to_upstream', 'to_upstream_types', 'to_flydsl']
