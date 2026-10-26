# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Public, versioned contract for FlyDSL ahead-of-time exports."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class AOTSchemaVersion:
    """A manifest schema version.

    Changes within one major version are additive. Consumers must ignore
    unknown fields and reject a manifest whose minimum consumer version is
    newer than the schema they implement.
    """

    major: int
    minor: int

    def to_dict(self) -> Dict[str, int]:
        return {"major": self.major, "minor": self.minor}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AOTSchemaVersion":
        return cls(major=int(value["major"]), minor=int(value["minor"]))


AOT_MANIFEST_SCHEMA = AOTSchemaVersion(major=3, minor=0)


@dataclass(frozen=True)
class AOTLayoutEntry:
    """One scalar field in a packed tensor-layout ABI slot."""

    kind: str
    dim: int
    offset: int
    bits: int
    signed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AOTLayoutEntry":
        return cls(
            kind=str(value["kind"]),
            dim=int(value["dim"]),
            offset=int(value["offset"]),
            bits=int(value["bits"]),
            signed=bool(value.get("signed", True)),
        )


@dataclass(frozen=True)
class AOTAbiSlot:
    """One host-storage entry in the packed ``void **args`` ABI."""

    arg_index: Optional[int]
    arg_name: Optional[str]
    kind: str
    ctype: str
    size: int
    alignment: int
    shape_dims: Optional[Tuple[int, ...]] = None
    stride_dims: Optional[Tuple[int, ...]] = None
    shape_bits: Optional[int] = None
    stride_bits: Optional[int] = None
    encoding: Optional[str] = None
    field: Optional[str] = None
    layout_entries: Tuple[AOTLayoutEntry, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for field_info in dataclasses.fields(self):
            value = getattr(self, field_info.name)
            if value is None:
                continue
            if field_info.name == "layout_entries":
                if value:
                    result[field_info.name] = [entry.to_dict() for entry in value]
                continue
            result[field_info.name] = list(value) if isinstance(value, tuple) else value
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AOTAbiSlot":
        return cls(
            arg_index=None if value.get("arg_index") is None else int(value["arg_index"]),
            arg_name=None if value.get("arg_name") is None else str(value["arg_name"]),
            kind=str(value["kind"]),
            ctype=str(value["ctype"]),
            size=int(value["size"]),
            alignment=int(value["alignment"]),
            shape_dims=(tuple(int(dim) for dim in value["shape_dims"]) if "shape_dims" in value else None),
            stride_dims=(tuple(int(dim) for dim in value["stride_dims"]) if "stride_dims" in value else None),
            shape_bits=(int(value["shape_bits"]) if "shape_bits" in value else None),
            stride_bits=(int(value["stride_bits"]) if "stride_bits" in value else None),
            encoding=str(value["encoding"]) if "encoding" in value else None,
            field=str(value["field"]) if "field" in value else None,
            layout_entries=tuple(AOTLayoutEntry.from_dict(entry) for entry in value.get("layout_entries", ())),
        )


@dataclass(frozen=True)
class AOTReturnABI:
    kind: str = "status"
    ctype: str = "int32"
    success: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AOTReturnABI":
        return cls(
            kind=str(value["kind"]),
            ctype=str(value["ctype"]),
            success=int(value["success"]),
        )


@dataclass(frozen=True)
class AOTProducer:
    flydsl_version: str
    backend: str
    arch: str
    host_triple: str
    abi_tag: str = "lp64"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AOTProducer":
        return cls(
            flydsl_version=str(value["flydsl_version"]),
            backend=str(value["backend"]),
            arch=str(value["arch"]),
            host_triple=str(value["host_triple"]),
            abi_tag=str(value.get("abi_tag", "lp64")),
        )


@dataclass(frozen=True)
class AOTLifecycle:
    init_symbol: str
    load_symbol: str
    unload_symbol: str

    def to_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AOTLifecycle":
        return cls(
            init_symbol=str(value["init_symbol"]),
            load_symbol=str(value["load_symbol"]),
            unload_symbol=str(value["unload_symbol"]),
        )


@dataclass(frozen=True)
class AOTLinkContract:
    libraries: Tuple[str, ...]
    options: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "libraries": [{"link_name": name} for name in self.libraries],
            "options": list(self.options),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AOTLinkContract":
        return cls(
            libraries=tuple(str(library["link_name"]) for library in value.get("libraries", ())),
            options=tuple(str(option) for option in value.get("options", ())),
        )


@dataclass(frozen=True)
class AOTManifest:
    """Stable description of one exported launcher entry."""

    schema_version: AOTSchemaVersion
    min_consumer_schema: AOTSchemaVersion
    producer: AOTProducer
    symbol: str
    metadata_symbol: str
    lifecycle: AOTLifecycle
    abi: Tuple[AOTAbiSlot, ...]
    return_abi: AOTReturnABI
    link: AOTLinkContract
    specialization_fingerprint: str
    capabilities: Tuple[str, ...]

    @property
    def module_init_symbol(self) -> str:
        return self.lifecycle.init_symbol

    @property
    def module_load_symbol(self) -> str:
        return self.lifecycle.load_symbol

    @property
    def module_unload_symbol(self) -> str:
        return self.lifecycle.unload_symbol

    @property
    def runtime_libraries(self) -> Tuple[str, ...]:
        return self.link.libraries

    @property
    def linker_flags(self) -> Tuple[str, ...]:
        return self.link.options

    @property
    def backend(self) -> str:
        return self.producer.backend

    @property
    def arch(self) -> str:
        return self.producer.arch

    @property
    def host_triple(self) -> str:
        return self.producer.host_triple

    @property
    def flydsl_version(self) -> str:
        return self.producer.flydsl_version

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version.to_dict(),
            "min_consumer_schema": self.min_consumer_schema.to_dict(),
            "producer": self.producer.to_dict(),
            "symbols": {
                "entry": self.symbol,
                "metadata": self.metadata_symbol,
                "lifecycle": self.lifecycle.to_dict(),
            },
            "entry": {
                "calling_convention": "packed_void_pp",
                "abi": [slot.to_dict() for slot in self.abi],
                "return_abi": self.return_abi.to_dict(),
            },
            "link": self.link.to_dict(),
            "specialization": {
                "fingerprint": self.specialization_fingerprint,
            },
            "capabilities": list(self.capabilities),
        }

    def require_compatible_consumer(self, consumer_schema: AOTSchemaVersion) -> None:
        if self.schema_version.major != consumer_schema.major:
            raise ValueError(
                "incompatible FlyDSL AOT manifest major version: "
                f"producer={self.schema_version.major}, "
                f"consumer={consumer_schema.major}"
            )
        minimum = self.min_consumer_schema
        if (minimum.major, minimum.minor) > (
            consumer_schema.major,
            consumer_schema.minor,
        ):
            raise ValueError(
                "FlyDSL AOT manifest requires consumer schema "
                f"{minimum.major}.{minimum.minor}, got "
                f"{consumer_schema.major}.{consumer_schema.minor}"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AOTManifest":
        symbols = value["symbols"]
        entry = value["entry"]
        if entry.get("calling_convention") != "packed_void_pp":
            raise ValueError("unsupported FlyDSL AOT calling convention: " f"{entry.get('calling_convention')!r}")
        return cls(
            schema_version=AOTSchemaVersion.from_dict(value["schema_version"]),
            min_consumer_schema=AOTSchemaVersion.from_dict(value["min_consumer_schema"]),
            producer=AOTProducer.from_dict(value["producer"]),
            symbol=str(symbols["entry"]),
            metadata_symbol=str(symbols["metadata"]),
            lifecycle=AOTLifecycle.from_dict(symbols["lifecycle"]),
            abi=tuple(AOTAbiSlot.from_dict(slot) for slot in entry["abi"]),
            return_abi=AOTReturnABI.from_dict(entry["return_abi"]),
            link=AOTLinkContract.from_dict(value["link"]),
            specialization_fingerprint=str(value["specialization"]["fingerprint"]),
            capabilities=tuple(str(item) for item in value.get("capabilities", ())),
        )


@dataclass(frozen=True)
class AOTExportResult:
    """Files and manifest produced by :meth:`CompiledFunction.export_aot`."""

    object_path: Path
    header_path: Path
    manifest_path: Path
    object_sha256: str
    header_sha256: str
    manifest: AOTManifest

    def to_sidecar_dict(self) -> Dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "files": {
                "object": {
                    "path": self.object_path.name,
                    "sha256": self.object_sha256,
                    "size": self.object_path.stat().st_size,
                },
                "header": {
                    "path": self.header_path.name,
                    "sha256": self.header_sha256,
                    "size": self.header_path.stat().st_size,
                },
            },
        }
