#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from find_blocks import delsum
from typing import Optional
import pathlib

parser = argparse.ArgumentParser(description='Show checksummed blocks in a file')
parser.add_argument('filename', type=pathlib.Path, help='File to show blocks from')
parser.add_argument('-m', '--model', type=str, help='Model to use for checksumming', required=True)
parser.add_argument('-g', '--gap', type=int, help='Block gap to display', required=True)
parser.add_argument('-e', '--end-pattern', type=str, default='', help="""A hex pattern (with "?" for digit wildcards) to match after end addresses.
                    Note that you probably want to have wildcards for the part where the checksum would be,
                    so if your checksum is 16-bit and you want to match the two bytes "be ef" after it,
                    you would use ????beef""")
parser.add_argument('-s', '--start-pattern', type=str, default='', help='A hex pattern (with "?" for digit wildcards) to match before start addresses.')
parser.add_argument('-l', '--longest-chain', action='store_true', help='Only display a longest chain of consecutive blocks')

args = parser.parse_args()

model: str = args.model
filename: str = args.filename
file: bytes = open(filename, "rb").read()
data = delsum(filename, model)[model]
gap: int = args.gap
longest_chain: bool = args.longest_chain

if len(args.start_pattern) % 2 != 0:
    raise ValueError("The start pattern must have an even number of characters.")
start_pattern = args.start_pattern
if len(args.end_pattern) % 2 != 0:
    raise ValueError("The end pattern must have an even number of characters.")
end_pattern = args.end_pattern

@dataclass
class Group:
    start: list[int]
    end: list[int]

def data_into_group_list(data) -> list[Group]:
    return [Group(item["start"], item["end"]) for item in data]

def next_addr(end_addr: int) -> int:
    return end_addr + 1 + gap

def prev_addr(start_addr: int) -> int:
    return start_addr - gap - 1

# Checks whether some bytes match a hex pattern that may contain "?" for wildcards.
def match_hex_pattern(pattern: str, text: bytes) -> bool:
    hex_str = text.hex()
    
    for p_char, h_char in zip(pattern, hex_str):
        if p_char != '?' and p_char.lower() != h_char:
            return False
    return True

def byte_position_matches(pos: int, pattern: str) -> bool:
    pattern_len = len(pattern) // 2
    if pos + pattern_len > len(file):
        return True
    
    return match_hex_pattern(pattern, file[pos:pos + pattern_len])

def map_start_to_group(data: list[Group]) -> dict[int, int]:
    starts = {}
    for (i, seg) in enumerate(data):
        for start in seg.start:
            starts[start] = i
    return starts

# keep track of all ends so we can quickly find the corresponding start
def map_end_to_group(data: list[Group]) -> dict[int, int]:
    ends = {}
    for (i, seg) in enumerate(data):
        for end in seg.end:
            ends[end] = i
    return ends

# write down all starts and ends that are adjacent to a gap
def collect_gapped_boundaries(data: list[Group], ends: dict[int, int]) -> tuple[list[set[int]], list[set[int]]]:
    block_starts: list[set[int]] = [set() for _ in range(len(data))]
    block_ends: list[set[int]] = [set() for _ in range(len(data))]

    for (i, seg) in enumerate(data):
        for start in seg.start:
            end_addr = prev_addr(start)
            if (byte_position_matches(start, start_pattern) and
                byte_position_matches(end_addr + 1, end_pattern) and
                end_addr in ends):
                block_starts[i].add(start)
                block_ends[ends[end_addr]].add(end_addr)

    return (block_starts, block_ends)

def find_groups_with_gaps(data: list[Group]) -> list[Group]:
    ends_map = map_end_to_group(data)
    (block_starts, block_ends) = collect_gapped_boundaries(data, ends_map)
    groups_with_gaps = []
    for (orig, starts, ends) in zip(data, block_starts, block_ends):
        if len(starts) == len(ends) == 0:
            continue
        all_starts = starts
        all_ends = ends
        orig_starts = orig.start
        orig_ends = orig.end
        # if one side was part of a gap, include all matching ends
        # from the other side
        if len(starts) > 0:
            minimum = min(starts)
            matches = lambda end: end > minimum and byte_position_matches(end + 1, end_pattern)
            all_ends = all_ends | {end for end in orig_ends if matches(end)}
        if len(ends) > 0:
            maximum = max(ends)
            matches = lambda start: start < maximum and byte_position_matches(start, start_pattern)
            all_starts = all_starts | {start for start in orig_starts if matches(start)} 
        if len(all_starts) == 0 or len(all_ends) == 0:
            continue
        groups_with_gaps.append(Group(sorted(all_starts), sorted(all_ends)))
    return groups_with_gaps

class Chain:
    count: int
    end: int
    next: Optional["Chain"]

    def __init__(self, end: int, next: Optional["Chain"] = None):
        self.end = end
        self.next = next
        if next:
            self.count = next.count + 1
        else:
            self.count = 0

    def is_better_match_than(self, other: Optional["Chain"]) -> bool:
        if other == None:
            return True
        return self.count > other.count

    def score(self, start: int) -> int:
        return self.count + 1
    
    def add_groups(self, groups: list[Group], start: int):
        cur: Optional["Chain"] = self
        while cur:
            groups.append(Group([start], [cur.end]))
            start = next_addr(cur.end)
            cur = cur.next

# We don't want to write the chain directly to the chain_records because of the following kind of situation:
# Assume gap = 2, and the following groups:
# group 0:  0:12
# group 1:  4,15:13,20
# group 2:  16:21
# When reaching end = 13, the loop will overwrite the previous entry in chain_records[1] because for end=13
# it has a higher count (as group 2 follows). 
# However for end = 12, it sees that the start after the gap at address 15 belongs to group 1 so doing it naively
# would connect to group 1, which currently contains a Chain with end=13, which is before 15 and would
# have negative length.
class DelayedWriter:
    queue: list[tuple[int, Chain]]
    chain_records: list[Optional[Chain]]
    min_end_distance: int

    def __init__(self,  chain_records: list[Optional[Chain]], min_length: int=1):
        self.chain_records = chain_records
        self.queue = []
        self.min_end_distance = gap + min_length
    
    def append(self, group: int, chain: Chain):
        self.queue.append((group, chain))
    
    def flush(self):
        for (group, chain) in self.queue:
            old_record = self.chain_records[group]
            if chain.is_better_match_than(old_record):
                self.chain_records[group] = chain
        self.queue = []
    
    def move_end(self, new_end: int):
        move_count = 0
        for (group, chain) in self.queue:
            if chain.end - new_end >= self.min_end_distance:
                old_record = self.chain_records[group]
                if chain.is_better_match_than(old_record):
                    self.chain_records[group] = chain
                move_count += 1
            else:
                break
        self.queue = self.queue[move_count:]


def find_longest_chain_index_from_group_candidates(group_candidates: list[Optional[Chain]], groups: list[Group]) -> int:
    maximum_idx = 0
    maximum_score = 0
    for (i, (candidate, group)) in enumerate(zip(group_candidates, groups)):
        if not candidate:
            continue
        start = min(group.start)
        new_score = candidate.score(start)
        if new_score > maximum_score:
            maximum_score = new_score
            maximum_idx = i
    return maximum_idx

def find_longest_chain(groups: list[Group]) -> list[Group]:
    if len(groups) == 0:
        return []
    starts = map_start_to_group(groups)
    ends = map_end_to_group(groups)
    chain_records: list[Optional[Chain]] = [None] * len(groups)
    end_list = sorted(ends.items(), reverse=True)
    queue = DelayedWriter(chain_records)
    for (end_addr, group_idx) in end_list:
        queue.move_end(end_addr)
        try:
            next_group_idx = starts[next_addr(end_addr)]
            next_chain = chain_records[next_group_idx]
            current_chain = Chain(end_addr, next_chain)
        except KeyError:
            current_chain = Chain(end_addr, None)
        queue.append(group_idx, current_chain)
    
    queue.flush()
    
    longest_chain_idx = find_longest_chain_index_from_group_candidates(chain_records, groups)
    record = chain_records[longest_chain_idx] 
    if not record:
        return []
    start = min(groups[longest_chain_idx].start)
    groups = []
    record.add_groups(groups, start)
    return groups

def find_blocks(data: list[Group]) -> list[Group]:
    blocks = find_groups_with_gaps(data)
    blocks.sort(key=lambda x: x.start)
    return blocks

def output_groups(groups: list[Group]):
    num_digits = len(hex(len(file))) - 2
    for group in groups:
        start_list = ','.join(f'{s:0{num_digits}x}' for s in group.start)
        end_list = ','.join(f'{e:0{num_digits}x}' for e in group.end)
        print(f'{start_list}:{end_list}')

    
groups = data_into_group_list(data)
blocks = find_blocks(groups)
if longest_chain:
    blocks = find_longest_chain(blocks)
output_groups(blocks)