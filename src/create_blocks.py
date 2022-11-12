from dataclasses import dataclass
from linsuffarr import SuffixArray
from linsuffarr import UNIT_BYTE
from typing import List


def create_token_array(_witness_token_lists):  # list of token lists per witness
    """Create token array (single list, with separator " # " between witnesses

    Returns:
        _token_array : flat list of tokens with ' #1 ' separators
        _token_membership_array : witness identifiers, same offsets as _token_array
        _token_witness_offset_array : offset of token in witness
        _token_ranges : list of tuples with (start, end+1) positions for each witness
            in _token_array
    """
    _token_array = []
    _token_membership_array = []
    _token_witness_offset_array = []
    _last_witness_offset = len(_witness_token_lists) - 1
    _token_ranges = []
    for _index, _witness_token_list in enumerate(_witness_token_lists):
        _witness_start = len(_token_array)
        _witness_end = len(_token_array) + len(_witness_token_list)
        _token_ranges.append((_witness_start, _witness_end), )
        _token_array.extend(_witness_token_list)
        for _token_offset, _token in enumerate(_witness_token_list):  # don't need enumerate, just len()
            _token_witness_offset_array.append(_token_offset)
        _token_membership_array.extend([_index for _token in _witness_token_list])
        if _index < _last_witness_offset:
            _separator = " #" + str(_index + 1) + " "
            _token_array.append(_separator)
            _token_membership_array.append(_separator)
            _token_witness_offset_array.append(-1)
    return _token_array, _token_membership_array, _token_witness_offset_array, _token_ranges


def _create_suffix_array(_token_array):
    """Create suffix array and LCP array

    Returns:
        SuffixArray object
        SuffixArray.SA is suffix array
        SuffixArray._LCP_values is LCP array
    """
    return SuffixArray(_token_array, unit=UNIT_BYTE)


@dataclass(unsafe_hash=True)
class Block:
    token_count: int
    start_position: int  # offset into suffix array (not into token array!)
    end_position: int  # start and end position give number of occurrences
    all_start_positions: []  # compute after blocks have been completed
    witnesses: set
    witness_count: int  # witness count for pattern, omitted temporarily because requires further computation
    frequency: int  # pattern count in whole witness set (may be more than once per witness), end_position - start_position + 1


@dataclass
class Lcp_interval_candidate:
    lcp_start_offset: int
    lcp_interval_token_count: int
    lcp_end_offset: int = -1


@dataclass
class LongestSequence:
    length: int
    witness_start_and_end: List[int]
