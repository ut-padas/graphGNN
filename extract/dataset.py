"""Sample pickles: one per stack, combined into the training dataset that train.py reads.

A sample is a plain namespace (types.SimpleNamespace) with the attributes listed in samples.py, so a pickle written
here loads with the standard library alone; train.py reads it with dill, which accepts it unchanged.
  save_samples, load_samples   one pickle = one sample list (one stack, or a combined dataset)
  combine_samples              the lists of several stacks as one list, in a stack order shuffled with a fixed seed:
                               train.py takes the last 5% of the list as the validation split, so the stacks must
                               not stay sorted by G and R (same shuffle as create_datasets.py, seed 35)
"""
import pickle, random


def save_samples(samples, path):
    with open(path, 'wb') as f:
        pickle.dump(list(samples), f, protocol=pickle.HIGHEST_PROTOCOL)


def load_samples(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def combine_samples(stacks, seed=35):
    """One list from the sample lists of several stacks, in a stack order shuffled with `seed` (None: the given order)."""
    stacks = list(stacks)
    if seed is not None:
        random.Random(seed).shuffle(stacks)
    return [s for stack in stacks for s in stack]
