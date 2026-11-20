# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Minimal read-only ELF64 little-endian parser.

Covers exactly what AOT export and loading need -- a shared library's
``DT_SONAME`` / ``DT_NEEDED`` entries and an object's global symbols and their
contents -- so neither depends on ``readelf``, ``ldd`` or ``nm`` being
installed. Malformed input raises ``ValueError``.
"""

import struct
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional, Tuple, Union

_EHDR = struct.Struct("<16sHHIQQQIHHHHHH")
_PHDR = struct.Struct("<IIQQQQQQ")
_SHDR = struct.Struct("<IIQQQQIIQQ")
_SYM = struct.Struct("<IBBHQQ")
_DYN = struct.Struct("<qQ")

_PT_LOAD = 1
_PT_DYNAMIC = 2
_SHT_SYMTAB = 2
_SHT_NOBITS = 8
_SHT_DYNSYM = 11
_DT_NULL = 0
_DT_NEEDED = 1
_DT_STRTAB = 5
_DT_SONAME = 14
_STB_GLOBAL = 1
_STB_WEAK = 2
_STV_DEFAULT = 0
_STV_PROTECTED = 3
_SHN_UNDEF = 0


class _DynamicInfo(NamedTuple):
    soname: Optional[str]
    needed: Tuple[str, ...]


class _Header(NamedTuple):
    phoff: int
    phentsize: int
    phnum: int
    shoff: int
    shentsize: int
    shnum: int


def _unpack(fmt: struct.Struct, data: bytes, offset: int) -> tuple:
    if offset < 0 or offset + fmt.size > len(data):
        raise ValueError("ELF structure out of file bounds")
    return fmt.unpack_from(data, offset)


def _read(source: Union[str, Path, bytes]) -> bytes:
    return source if isinstance(source, bytes) else Path(source).read_bytes()


def _header(data: bytes) -> _Header:
    if data[:4] != b"\x7fELF" or len(data) < _EHDR.size:
        raise ValueError("not an ELF file")
    if data[4] != 2 or data[5] != 1:
        raise ValueError("only little-endian ELF64 is supported")
    fields = _unpack(_EHDR, data, 0)
    phoff, shoff = fields[5], fields[6]
    phentsize, phnum, shentsize, shnum = fields[9], fields[10], fields[11], fields[12]
    if (phnum and phentsize != _PHDR.size) or (shnum and shentsize != _SHDR.size):
        raise ValueError("unexpected ELF header entry sizes")
    return _Header(phoff, phentsize, phnum, shoff, shentsize, shnum)


def _cstr(data: bytes, offset: int) -> str:
    end = data.find(b"\0", offset) if 0 <= offset < len(data) else -1
    if end < 0:
        raise ValueError("string table offset out of range")
    return data[offset:end].decode("utf-8")


def _dynamic_info(source: Union[str, Path, bytes]) -> _DynamicInfo:
    """Return the ``DT_SONAME`` and ``DT_NEEDED`` entries of a shared library.

    Read from the ``PT_DYNAMIC`` segment, which the dynamic loader itself uses,
    so it also works for libraries whose section headers were stripped.
    """
    data = _read(source)
    hdr = _header(data)
    segments = [_unpack(_PHDR, data, hdr.phoff + i * _PHDR.size) for i in range(hdr.phnum)]
    dynamic = [seg for seg in segments if seg[0] == _PT_DYNAMIC]
    if not dynamic:
        return _DynamicInfo(None, ())

    # (p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align)
    _type, _flags, dyn_offset, _vaddr, _paddr, dyn_size, _memsz, _align = dynamic[0]
    entries = []
    for off in range(dyn_offset, dyn_offset + dyn_size, _DYN.size):
        tag, val = _unpack(_DYN, data, off)
        if tag == _DT_NULL:
            break
        entries.append((tag, val))

    strtab_vaddr = next((val for tag, val in entries if tag == _DT_STRTAB), None)
    if strtab_vaddr is None:
        raise ValueError("dynamic segment has no DT_STRTAB")
    for p_type, _flags, offset, vaddr, _paddr, filesz, _memsz, _align in segments:
        if p_type == _PT_LOAD and vaddr <= strtab_vaddr < vaddr + filesz:
            strtab = strtab_vaddr - vaddr + offset
            break
    else:
        raise ValueError("DT_STRTAB is not inside a loadable segment")

    soname = None
    needed = []
    for tag, val in entries:
        if tag == _DT_NEEDED:
            needed.append(_cstr(data, strtab + val))
        elif tag == _DT_SONAME:
            soname = _cstr(data, strtab + val)
    return _DynamicInfo(soname, tuple(needed))


def _sections(data: bytes) -> List[tuple]:
    hdr = _header(data)
    # (sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize)
    return [_unpack(_SHDR, data, hdr.shoff + i * _SHDR.size) for i in range(hdr.shnum)]


def _symbols(data: bytes, sections: List[tuple], dynamic: bool) -> Iterator[Tuple[str, tuple]]:
    """Yield ``(name, (st_name, st_info, st_other, st_shndx, st_value, st_size))`` per symbol."""
    for _name, sh_type, _flags, _addr, offset, size, link, _info, _align, _entsize in sections:
        if sh_type != (_SHT_DYNSYM if dynamic else _SHT_SYMTAB):
            continue
        if link >= len(sections):
            raise ValueError("symbol table links to a missing string table")
        strtab = sections[link][4]
        for off in range(offset, offset + size, _SYM.size):
            sym = _unpack(_SYM, data, off)
            yield _cstr(data, strtab + sym[0]), sym


def _global_symbols(
    source: Union[str, Path, bytes],
    *,
    dynamic: bool = False,
    defined: Optional[bool] = None,
    externally_visible: bool = False,
) -> List[str]:
    """Return selected global and weak symbols from an ELF symbol table."""
    data = _read(source)
    names = []
    for name, (_st_name, st_info, st_other, st_shndx, _value, _size) in _symbols(data, _sections(data), dynamic):
        if st_info >> 4 not in (_STB_GLOBAL, _STB_WEAK):
            continue
        is_defined = st_shndx != _SHN_UNDEF
        if defined is not None and is_defined != defined:
            continue
        if externally_visible and st_other & 0x3 not in (_STV_DEFAULT, _STV_PROTECTED):
            continue
        names.append(name)
    return names


def _defined_global_symbols(source: Union[str, Path, bytes], *, dynamic: bool = False) -> List[str]:
    """Return the defined, externally visible symbols of an ELF file.

    Reads ``.symtab`` (relocatable objects), or ``.dynsym`` -- the symbols a
    shared library exports -- when ``dynamic`` is true.
    """
    return _global_symbols(source, dynamic=dynamic, defined=True, externally_visible=True)


def _symbol_contents(source: Union[str, Path, bytes], name: str) -> bytes:
    """Return the bytes a relocatable object's defined symbol ``name`` covers, as stored in the file."""
    data = _read(source)
    sections = _sections(data)
    for sym_name, (_st_name, _info, _other, st_shndx, value, size) in _symbols(data, sections, dynamic=False):
        if sym_name != name or st_shndx == _SHN_UNDEF:
            continue
        if st_shndx >= len(sections) or sections[st_shndx][1] == _SHT_NOBITS:
            raise ValueError(f"symbol {name} has no contents in the file")
        start = sections[st_shndx][4] + value
        if value + size > sections[st_shndx][5] or start + size > len(data):
            raise ValueError(f"symbol {name} is out of file bounds")
        return data[start : start + size]
    raise ValueError(f"symbol {name} is not defined")
