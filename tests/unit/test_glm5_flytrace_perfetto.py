# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import gzip
import json
from collections import Counter

import pytest

from kernels.mla_moe_layer.tools.flytrace_perfetto import export_hierarchical_pftrace

pytestmark = pytest.mark.l0_backend_agnostic


def _varint(data, offset):
    value = shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7


def _fields(data):
    offset = 0
    while offset < len(data):
        key, offset = _varint(data, offset)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, offset = _varint(data, offset)
        elif wire == 2:
            length, offset = _varint(data, offset)
            value = data[offset : offset + length]
            offset += length
        else:
            raise AssertionError(f"unexpected protobuf wire type {wire}")
        yield field, value


def _synthetic_trace():
    marks = []
    for start, end in ((1.0, 2.0), (3.0, 5.0)):
        marks.extend(
            [
                dict(ph="i", name="qkv_a/start", pid=1, tid=1, ts=start, args=dict(payload=7)),
                dict(ph="i", name="qkv_a/publish", pid=1, tid=1, ts=end, args=dict(payload=7)),
            ]
        )
    return dict(
        traceEvents=[
            dict(
                ph="M",
                name="thread_name",
                pid=1,
                tid=1,
                args=dict(name="indexed_mla_moe_kernel_0 Block (0, 0, 0) / wave 0"),
            ),
            *marks,
            dict(ph="X", name="gating", pid=1, tid=1, ts=1.25, dur=0.25, args=dict(payload=0)),
            dict(ph="X", name="glm5_mla_moe", pid=1, tid=1, ts=0.5, dur=5.0, args=dict(payload=3)),
        ]
    )


def test_hierarchical_pftrace_has_balanced_nested_slices(tmp_path):
    source = tmp_path / "flytrace.json"
    target = tmp_path / "flytrace.pftrace.gz"
    source.write_text(json.dumps(_synthetic_trace()))

    stats = export_hierarchical_pftrace(source, target, rank=2)
    assert stats == {"pftrace_tracks": 10, "pftrace_slices": 9, "pftrace_instants": 0}

    packets = [value for field, value in _fields(gzip.open(target, "rb").read()) if field == 1]
    descriptors = []
    event_types = Counter()
    event_names = Counter()
    for packet in packets:
        for field, value in _fields(packet):
            if field == 60:
                descriptor = dict(_fields(value))
                descriptors.append(
                    (
                        descriptor[1],
                        descriptor[2].decode(),
                        descriptor.get(5),
                    )
                )
            elif field == 11:
                event = dict(_fields(value))
                event_types[event[9]] += 1
                event_names[event[23].decode()] += 1

    descriptor_by_name = {name: (uuid, parent) for uuid, name, parent in descriptors}
    assert descriptor_by_name["Rank 2"][1] == descriptor_by_name["Root"][0]
    assert descriptor_by_name["GLM5 Layer 3"][1] == descriptor_by_name["Rank 2"][0]
    assert descriptor_by_name["Stage Breakdown"][1] == descriptor_by_name["GLM5 Layer 3"][0]
    assert descriptor_by_name["qkv_a"][1] == descriptor_by_name["Stage Breakdown"][0]
    assert descriptor_by_name["gating"][1] == descriptor_by_name["Stage Breakdown"][0]
    assert descriptor_by_name["Block (0, 0, 0)"][1] == descriptor_by_name["indexed_mla_moe_kernel_0"][0]
    assert descriptor_by_name["StackedRanges"][1] == descriptor_by_name["Wave00"][0]
    assert event_types == {1: 9, 2: 9}
    assert event_names["FlyDSL · qkv_a[7]"] == 4  # Two distinct tasks reuse the same payload.
    assert event_names["FlyDSL · start → publish"] == 4
    assert event_names["FlyDSL · gating"] == 2
    assert event_names["FlyDSL · gating[sample=0]"] == 2
    assert event_names["FlyDSL · kernel_e2e"] == 2


def test_hierarchical_pftrace_rejects_incomplete_input(tmp_path):
    source = tmp_path / "flytrace.json"
    source.write_text(json.dumps({"traceEvents": []}))
    with pytest.raises(ValueError, match="no wave events"):
        export_hierarchical_pftrace(source, tmp_path / "trace.pftrace.gz", rank=0)
    with pytest.raises(ValueError, match="must end in .pftrace.gz"):
        export_hierarchical_pftrace(source, tmp_path / "trace.json", rank=0)


def test_hierarchical_pftrace_accepts_legacy_cta_labels(tmp_path):
    trace = _synthetic_trace()
    trace["traceEvents"][0]["args"]["name"] = "legacy_kernel CTA (1, 2, 3) / wave 4"
    source = tmp_path / "flytrace.json"
    target = tmp_path / "flytrace.pftrace.gz"
    source.write_text(json.dumps(trace))

    export_hierarchical_pftrace(source, target, rank=0)

    packets = [value for field, value in _fields(gzip.open(target, "rb").read()) if field == 1]
    track_names = {
        dict(_fields(descriptor))[2].decode()
        for packet in packets
        for field, descriptor in _fields(packet)
        if field == 60
    }
    assert "Block (1, 2, 3)" in track_names
    assert not any("CTA" in name for name in track_names)
