#!/usr/bin/env python3
# In some file formats, the data is divided into blocks, with each block
# having a checksum appended to it. This means that, given a list of
# regions with valid checksums, in those formats we will find a higher
# count of regions where the end of a region is a fixed distance from the
# start of another region (the block gap).
# This script takes a list of checksum algorithms and a list of files and
# attempts to find the most common block gaps in the files, using the
# geometric mean of the number of blocks with a given gap size to combine
# the scores from multiple files.
import numpy as np
from scipy import signal
import subprocess
import os
import sys
import json
import pathlib
import argparse

try:
    delsum_path = os.environ['DELSUM_PATH']
except KeyError:
    delsum_path = 'delsum'

# Takes a file and a model and returns the delsum output as a dictionary
# with models as keys and returning a list of start/end pairs of lists
def delsum(file, model):
    args = [delsum_path, 'part', '-j', '-s', '-p', '-t0']
    if isinstance(model, pathlib.Path):
        args.extend(['-M', model])
    else:
        args.extend(['-m', model])
    args.append(file)

    output = subprocess.run(args, capture_output=True, text=True)
    if output.returncode != 0:
        raise ValueError(f'Error running delsum: {output.stderr}')
    return json.loads(output.stdout)

# Returns a list containing, for each gap size corresponding to the current
# index, the number of blocks that have a gap of that size
def correlate(size, file_model_data):
    starts = np.zeros(size, dtype=np.float32)
    ends = np.zeros(size, dtype=np.float32)
    for segs in file_model_data:
        for start in segs["start"]:
            starts[start] = 1
        for end in segs["end"]:
            ends[end] = 1
    res = np.round(signal.correlate(starts, ends, mode='full'))
    # the output of delsum are inclusive ranges, so we effectively
    # subtract 1 from the gap sizes here to make them exclusive
    # because the middle is at size - 1
    return res[size:]

def find_blocks_for_model(sizes, model_data, top):
    # calculate the geometric mean of the scores
    scores = np.ones(np.max(sizes) - 1, dtype=np.float64)
    for (size, data) in zip(sizes, model_data):
        # make sure that zeros are not included in the geometric mean
        scores[:size - 1] *= correlate(size, data) + 1
    scores = np.power(scores, 1/len(sizes)) - 1
    top_idx = np.argsort(scores)[::-1][:top]
    return (top_idx, scores[top_idx])

# Given a list of files and a list of models, score each gap width, combining
# the scores from multiple files using the geometric mean and then return the
# top `top` gap widths for each model
def find_blocks(files, model, top):
    sizes = []
    data = []
    for file in files:
        try:
            size = os.path.getsize(file)
            data.append(delsum(file, model))
            sizes.append(size)
        except Exception as e:
            print(f'Error processing {file}: {e}, skipping...', file=sys.stderr)
    models = list(data[0].keys())
    scores = []
    for model in models:
        scores.append(find_blocks_for_model(sizes, [d[model] for d in data], top))
    top_scores = [s[1][0] for s in scores]
    idx = np.argsort(top_scores)[::-1]
    scores_sorted = [scores[i] for i in idx]
    models_sorted = [models[i] for i in idx]
    return (models_sorted, scores_sorted)
 
def main():
    parser = argparse.ArgumentParser(description='Find checksummed blocks in a file')
    parser.add_argument('filenames', type=pathlib.Path, help='Files to search for blocks', nargs='+')
    parser.add_argument('-m', '--model', type=str, help='Model to use for checksumming')
    parser.add_argument('-M', '--model-file', type=pathlib.Path,
                        help='File containing models to use for checksumming')
    parser.add_argument('-t', '--top', type=int, default=3,
                        help='Number of top block gaps to display')

    args = parser.parse_args()
    
    match (args.model, args.model_file):
        case (model, None) | (None, model):
            (models, scores) = find_blocks(args.filename, model, args.top)
        case (None, None):
            raise ValueError('Must specify either a model or a model file')
        case (_, _):
            raise ValueError('Must specify only one of model or model file')
    
    for (model, score) in zip(models, scores):
        print(f'Model: {model}')
        for (idx, s) in zip(*score):
            print(f'Block gap: {idx}, Score: {s:.3f}')
    
if __name__ == '__main__':
    main()