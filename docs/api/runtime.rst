Stable runtime helpers
======================

FlyDSL exposes two stable host-side helpers for selecting architecture-specific
kernel paths. Import them from ``flydsl.runtime.device``:

.. code-block:: python

   from flydsl.runtime.device import get_rocm_arch, is_rdna_arch

``get_rocm_arch()``
-------------------

Return a lower-case ROCm architecture name such as ``gfx942`` or ``gfx1201``.
The helper checks, in order:

1. ``FLYDSL_GPU_ARCH``;
2. ``HSA_OVERRIDE_GFX_VERSION``;
3. the first GPU reported by ``rocm_agent_enumerator``;
4. ``gfx942`` as a best-effort fallback when detection is unavailable.

An override may use either the normal ``gfx*`` spelling or ROCm's dotted form;
for example, ``9.4.2`` is normalized to ``gfx942``.

``is_rdna_arch(arch=None)``
---------------------------

Return whether an architecture belongs to an RDNA family. Passing ``None``
uses ``get_rocm_arch()``. The recognized prefixes are ``gfx10``, ``gfx11``, and
``gfx120``. ``gfx1250`` is wave32 CDNA5 and is intentionally not classified as
RDNA.

.. code-block:: python

   arch = get_rocm_arch()
   uses_rdna_instructions = is_rdna_arch(arch)

Only these two runtime paths are part of the stable API contract. Runtime
implementation, registration, and compatibility machinery is intentionally
omitted from the public reference. See :doc:`../api_stability` for the exact
stability rules and :doc:`../architecture_guide` for compiler, cache, and debug
configuration.
