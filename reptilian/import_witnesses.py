from typing import List
import re


def _tokenize_witnesses(witness_strings: List[str]):  # one string per witness
    """Return list of witnesses, each represented by a list of tokens"""
    # TODO: handle punctuation, upper- vs lowercase
    witnesses = []
    for witness_string in witness_strings:
        # witness_tokens = witness_string.split()
        witness_tokens = re.findall(r'\w+\s*|\W+', witness_string)
        witness_tokens = [token.strip().lower() for token in witness_tokens]
        witnesses.append(witness_tokens)
    return witnesses


def import_witnesses():
    filenames = ['../darwin/darwin1859.txt', '../darwin/darwin1860.txt', '../darwin/darwin1861.txt',
                 '../darwin/darwin1866.txt', '../darwin/darwin1869.txt', '../darwin/darwin1872.txt']
    sigla = ["".join(['w', str(i)]) for i in range(len(filenames))]
    first_paragraph = 0
    last_paragraph = 100
    raw_data_dict = {}
    for siglum, filename in zip(sigla, filenames):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line for line in lines if line != '\n']
            raw_data_dict[siglum] = " ".join(lines[first_paragraph: last_paragraph])
    witness_sigla = [key for key in raw_data_dict.keys()]
    witnesses = _tokenize_witnesses([value for value in raw_data_dict.values()])  # strings
    return witness_sigla, witnesses
