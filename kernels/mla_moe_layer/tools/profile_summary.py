"""Summarize saved diagnostic traces; timestamps are 100 MHz on MI355X."""

import argparse
import csv
import sys

import torch


def rows(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    ticks = data["ticks"].double()
    start = ticks[:, 0].min()
    if "points" in data:
        for column, name in enumerate(data["points"].values()):
            values = ticks[:, column]
            values = (values[values > 0] - start) / 100
            if values.numel():
                yield name, values.numel(), values.min().item(), values.median().item(), values.max().item()
    else:
        offset = 0
        for name, count in data["stages"]:
            stage = ticks[offset : offset + count]
            offset += count
            for column, event in ((0, "start"), (1, "hint"), (2, "ready"), (3, "computed"), (4, "publish")):
                values = stage[:, column]
                values = (values[values > 0] - start) / 100
                if values.numel():
                    yield f"{name}_{event}", values.numel(), values.min().item(), values.median().item(), values.max().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    args = parser.parse_args()
    writer = csv.writer(sys.stdout)
    writer.writerow(("event", "ctas", "first_us", "median_us", "last_us"))
    for name, count, first, median, last in rows(args.trace):
        writer.writerow((name, count, f"{first:.3f}", f"{median:.3f}", f"{last:.3f}"))
