#!/usr/bin/env python3
import argparse
from find_blocks import delsum
import pathlib

parser = argparse.ArgumentParser(description='Show checksummed blocks in a file')
parser.add_argument('filename', type=pathlib.Path, help='File to show blocks from')
parser.add_argument('-m', '--model', type=str, help='Model to use for checksumming', required=True)
parser.add_argument('-g', '--gap', type=int, help='Block gap to display', required=True)

args = parser.parse_args()

model = args.model
filename = args.filename
data = delsum(filename, model)[model]
gap = args.gap

# keep track of all ends so we can quickly find the corresponding start
ends = {}
for (i, seg) in enumerate(data):
    for end in seg["end"]:
        ends[end] = i


# write down all starts and ends that are adjacent to a gap
block_starts = [set() for _ in range(len(data))]
block_ends = [set() for _ in range(len(data))]

for (i, seg) in enumerate(data):
    for start in seg["start"]:
        end_addr = start - (gap + 1)
        if end_addr in ends:
            block_starts[i].add(start)
            block_ends[ends[end_addr]].add(end_addr)

num_digits = len(hex(max(ends.keys()))) - 2

for (orig, starts, ends) in zip(data, block_starts, block_ends):
    if len(starts) == len(ends) == 0:
        continue
    all_starts = starts
    all_ends = ends
    orig_starts = orig["start"]
    orig_ends = orig["end"]
    # if one side was part of a gap, include all matching ends
    # from the other side, but only if only one side is part of a gap
    if len(starts) > 0 and len(ends) == 0:
        minimum = min(starts)
        all_ends = all_ends | {end for end in orig_ends if end > minimum}
    elif len(ends) > 0 and len(starts) == 0:
        maximum = max(ends)
        all_starts = all_starts | {start for start in orig_starts if start < maximum}
    start_list = ','.join(f'{s:0{num_digits}x}' for s in sorted(all_starts))
    end_list = ','.join(f'{e:0{num_digits}x}' for e in sorted(all_ends))
    print(f'{start_list}:{end_list}')
