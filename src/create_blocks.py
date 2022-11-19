from dataclasses import dataclass
from linsuffarr import SuffixArray
from linsuffarr import UNIT_BYTE
from typing import List
from bisect import bisect_right
from heapq import *  # priority heap, https://docs.python.org/3/library/heapq.html
import numpy as np


def create_token_array(_witness_token_lists):  # list of token lists per witness
    """Create token array (single list, with separator " # " between witnesses)

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


def create_suffix_array(_token_array):
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
    frequency: int  # pattern count in witnesses (may be more than once per witness), end_position - start_position + 1


@dataclass
class LcpIntervalCandidate:
    lcp_start_offset: int
    lcp_interval_token_count: int
    lcp_end_offset: int = -1


@dataclass
class LongestSequence:
    length: int
    witness_start_and_end: List[int]


def check_for_depth_and_repetition(_suffix_array, _token_membership_array, _lcp_interval: LcpIntervalCandidate,
                                   _witness_count: int) -> bool:
    """Write a docstring someday

    Number of prefixes >= total number of witnesses
    Accumulate set of witness sigla for prefixes
    if:
        no witness occurs more than once, return True to keep this block
    else:
        return False
    """
    block_instance_count = _lcp_interval.lcp_end_offset - _lcp_interval.lcp_start_offset + 1
    if block_instance_count != _witness_count:
        return False
    else:
        _witnesses_found = []
        for _lcp_interval_item_offset in range(_lcp_interval.lcp_start_offset, _lcp_interval.lcp_end_offset + 1):
            _token_position = _suffix_array.SA[_lcp_interval_item_offset]  # point from prefix to suffix array position
            _witness_siglum = _token_membership_array[
                _token_position]  # point from token array position to witness identifier
            if _witness_siglum in _witnesses_found:
                return False
            else:
                _witnesses_found.append(_witness_siglum)
        return True


def create_blocks(_suffix_array, _token_membership_array, _witness_count):
    """Write a docstring someday

    Look at changes in length of LCP array
    Initial value is 0 or -1 because it's a comparison with previous, and first has no previous
    Next value is number of tokens shared with previous
    Exact length doesn't matter, but if it changes, new pattern:
        If it stays the same, take note but do nothing yet; it means that the pattern repeats
        No change for a while, then goes to 0:
            Number of repetitions plus 1, e.g., 5 5 5 0 = 4 instances of 5
            Once it changes to 0, we've seen complete pattern
        Changer to smaller means hidden, deeper block
        Changes to longer means ???
    """
    _accumulator = []  # lcp positions (not values) since most recent 0
    _frequent_sequences = []  # lcp intervals to be considered for mfs
    _lcp_array = _suffix_array._LCP_values
    #
    # lcp value
    # if == 0 it's a new interval, so:
    #   1. if there is already an accumulation, commit (process) it
    #      "committing the buffer" means checking for repetition and depth
    #          if it passes check: store in mfs list
    #          otherwise throw it away
    #   2. clear buffer (accumulator) and begin accumulating new buffer with the new offset with 0 value
    # otherwise it isn't zero, so there must be a buffer in place, so add to it (for now)
    for _offset, _value in enumerate(_lcp_array):
        if not _accumulator and _value == 0:  # if accumulator is empty and new value is 0, do nothing
            continue
        elif not _accumulator:  # accumulator is empty and new value is non-zero, so begin new accumulator
            _accumulator.append(LcpIntervalCandidate(lcp_start_offset=_offset - 1, lcp_interval_token_count=_value))
        elif _value > _accumulator[-1].lcp_interval_token_count:  # new interval, so add to accumulator and continue
            _accumulator.append(LcpIntervalCandidate(lcp_start_offset=_offset - 1, lcp_interval_token_count=_value))
        elif _value == _accumulator[-1].lcp_interval_token_count:  # same block as before, so do nothing
            continue
        else:  # new value is less than top of accumulator, so pop everything that is higher
            # Positions in lcp array and suffix array coincide:
            #   The lcp array value is the length of the sequence
            #   The suffix array value is the start position of the sequence
            # Assume accumulator values (offsets into lcp array) point to [3, 6] and new value is 4, so:
            #   First: Pop pointer to 6 (length value in lcp array), store in frequent_sequences
            #   Second: Push new pointer to same position in lcp array, but change value in lcp array to 4
            while _accumulator and _accumulator[-1].lcp_interval_token_count > _value:
                # Create pointer to last closed block that is not filtered (like frequent_sequences)
                _newly_closed_block = _accumulator.pop()
                _newly_closed_block.lcp_end_offset = _offset - 1
                if check_for_depth_and_repetition(_suffix_array, _token_membership_array, _newly_closed_block,
                                                  _witness_count):
                    _frequent_sequences.append(
                        [_newly_closed_block.lcp_start_offset, _newly_closed_block.lcp_end_offset,
                         _newly_closed_block.lcp_interval_token_count])
            # There are three options:
            #   1. there is content in the accumulator and latest value is not 0
            #   2. accumulator is empty and latest value is 0
            #   3. accumulator is empty and latest value is not 0
            # (the fourth logical combination, content in the accumulator and 0 value, cannot occur
            #     because a 0 value will empty the accumulator)
            if _value > 0 and (not _accumulator or _accumulator[-1].lcp_interval_token_count != _value):
                _accumulator.append(LcpIntervalCandidate(lcp_start_offset=_newly_closed_block.lcp_start_offset,
                                                         lcp_interval_token_count=_value))
    # End of lcp array; run through any residual accumulator values
    while _accumulator:
        _newly_closed_block = _accumulator.pop()
        _newly_closed_block.lcp_end_offset = len(_lcp_array) - 1
        if check_for_depth_and_repetition(_suffix_array, _token_membership_array, _newly_closed_block, _witness_count):
            _frequent_sequences.append([_newly_closed_block.lcp_start_offset, len(_lcp_array) - 1,
                                        _newly_closed_block.lcp_interval_token_count])
    return _frequent_sequences


def find_longest_sequences(_frequent_sequences, _suffix_array):
    """Returns largest blocks from list of all blocks

    Remove embedded prefixes"""
    _largest_blocks = {}  # key is token end position, value is (length, [witness-start-positions])
    for _frequent_sequence in _frequent_sequences:
        _length = _frequent_sequence[2]
        _suffix_array_values = [_suffix_array.SA[i] for i in range(_frequent_sequence[0], _frequent_sequence[1] + 1)]
        _token_end_position = min(_suffix_array_values) + _length  # token end position for first witness
        if _token_end_position not in _largest_blocks:  # first block with this end position, so create new key
            _largest_blocks[_token_end_position] = (_length, sorted(_suffix_array_values))
        else:  # if new block is longer, replace old one with same key
            if _length > _largest_blocks[_token_end_position][0]:
                _largest_blocks[_token_end_position] = (_length, sorted(_suffix_array_values))
    return _largest_blocks


def prepare_for_beam_search(_witness_count, _token_membership_array, _largest_blocks):
    # block_offsets_by_witness: list of lists holds sorted start offsets per witness (offsets are into global token array)
    # witness_offsets_to_blocks: dictionary points from start offsets to blocks
    # score_by_block: number of tokens placed or skipped if block is placed
    # Beam search requires us, given an offset in a witness, to find the next block. We do
    #   that by looking up the value in block_offsets_by_witness and then using that value
    #   to retrieve the block key from witness_offsets_to_blocks
    # Lookup in the list of lists is:
    #   block_offsets_by_witness[witness_number][bisect_right(block_offsets_by_witness[witness_number], most_recent_offset_in_witness)]
    # (See: https://www.geeksforgeeks.org/python-find-smallest-element-greater-than-k/)
    # FIXME: traverse largest_blocks only once and add values for all witnesses in same pass
    _witness_count = _witness_count
    _block_offsets_by_witness = []
    _witness_offsets_to_blocks = {}
    _first_token_offset_in_block_by_witness = []  # only tokens in blocks
    _first_absolute_token_by_witness = []  # all tokens, whether in block or not
    for i in range(_witness_count):
        _first_token_offset_in_block_by_witness.append(_token_membership_array.index(i))
        # Score = number of tokens either placed or skipped (we don't care which)
        # Low score is best because it leaves the highest potential
        # NB: The name "score" seems to imply that higher is better, and the
        #   opposite is the case here. Rename the variable?
        # NB: High potential is paramount during beam search, but should the
        #   difference between placed and skip matter at a later stage? Or
        #   does placing more blocks (more tiers) take care of that?
        _score_by_block = {}
        for i in range(_witness_count):
            _witness_offset_list = []
            for _key, _value in _largest_blocks.items():
                _witness_offset_list.append(_value[1][i])
                _witness_offsets_to_blocks[_value[1][i]] = _key
            _witness_offset_list.sort()
            _block_offsets_by_witness.append(_witness_offset_list)
    for i in range(_witness_count):
        _first_absolute_token_by_witness.append(_token_membership_array.index(i))
    for _key, _value in _largest_blocks.items():
        # to determine number of tokens that will have been placed or skipped
        #   after placing block:
        #       matrix-subtract first_token_offset_by_witness from value[1]
        #       add witness_count * value[0] (to account for block length)
        #   key by block key, value is score
        _differences = [x - y for x, y in zip(_value[1], _first_token_offset_in_block_by_witness)]
        _score = sum(_differences) + _witness_count * _value[0]
        _score_by_block[_key] = _score
    return _block_offsets_by_witness, _witness_offsets_to_blocks, \
           _first_token_offset_in_block_by_witness, _first_absolute_token_by_witness, \
           _score_by_block


@dataclass(order=True, frozen=True, eq=True)  # heapqueue is priority queue, so requires comparison
class BeamOption:
    score: int
    path: tuple  # path through sequence of blocks leading to current BeamOption


def perform_beam_search_step(_witness_count,
                             _largest_blocks,
                             _block_offsets_by_witness,
                             _witness_offsets_to_blocks,
                             _score_by_block,
                             _beam_options=[BeamOption(score=0, path=())],
                             _beta=3):
    # TODO: The witness count should be a global constant
    # print("New tier with " + str(len(beam_options)) + " beam options")
    _new_options = []  # candidates for next tier
    _finished_options = []
    for _beam_option in _beam_options:
        # ###
        # 2022-09-06
        # Three possibilities for an individual beam option:
        # 1. Option leads to new option
        # 2. Option is finished
        # 3. No new option but option isn't finished (transposition)
        # NB: We check each witness in the beam option, and if any witness
        #     raises an IndexError, the whole block cannot be advanced and
        #     is finished. (This is true because of our constraints: every
        #     block is a) full-depth and b) no repetition.)
        #
        # What to do:
        #
        # 1. Perform the bisect for each witness based on the head of the
        #    path of the current beam option. This returns an offset into
        #    the witness-specific list of block offsets. Initialize a
        #    counter to 0.
        # 2. Using the initial offsets returned by the bisect operation that
        #    we performed in step #1 (and never perform again) for each witness
        #    plus the counter (which we will increment if needed in the inner
        #    loop), check that next option for each witness. There are three
        #    possibilities for each counter value (over the entire witness group):
        #    a) If the next block (returned by this method) would overrun
        #       for any witness, it will overrun for all witnesses, so the
        #       beam option can be added to the finished list and we exit the
        #       outer loop (the one that processes the beam option).
        #    b) If the next block is a viable option, add it to the options and
        #       check the next witness within this same inner loop instance
        #       because in case of transposition different blocks will suggest
        #       different next blocks, all of which could be viable options.
        #       This ends the processing for that beam option.
        #    c) If we don't find any viable next option and don't overrun for
        #       any witness, increment the counter and replay step #2 (inner
        #       loop).
        # Exit condition: Eventually we either find a viable option or overrun.
        #
        # TODO: How should we implement this to terminate the correct loop in
        # the right place? For? While? Generator? For and while start with the
        # outer loop and work inward; with a generator we start with the inner
        # and work outward.
        # ###
        _new_finished_option_check = False
        _new_viable_option_check = 0
        _counter = 0
        while True:
            for i in range(_witness_count):  # advance for each witness in turn
                if not _beam_option.path:  # path is empty only for initial state at tier 0
                    _last_offset = -1  # NB: same for all witnesses, and not 0, which will break for witness 0
                else:
                    _last_offset = _largest_blocks[_beam_option.path[0]][1][i]
                try:
                    _next_offset = bisect_right(_block_offsets_by_witness[i], _last_offset)
                    _next_value = _block_offsets_by_witness[i][_next_offset + _counter]
                    _next_block = _witness_offsets_to_blocks[_next_value]  # find that next block to get its length
                    # would any witness pointer move backwards?
                    # perform matrix subtraction; if signs differ, there are items that move in opposite directions
                    # first option cannot be transposed, so accept it automatically
                    if (not _beam_option.path) or (len(set([np.sign(x - y) for x, y in
                                                            zip(_largest_blocks[_next_block][1],
                                                                _largest_blocks[_beam_option.path[0]][1])])) == 1):
                        _new_score = _score_by_block[_next_block]  # accounts for all witnesses
                        # concatenate tuples with a +;  most recent first (for priority heap)
                        _new_options.append(BeamOption(score=_new_score, path=((_next_block,) + _beam_option.path)))
                        _new_viable_option_check += 1
                    else:
                        continue
                        # print('Transposition detected for beam option:', beam_option)
                except IndexError:  # we've gone as far as we can with this path
                    _new_finished_option_check = True
                    _finished_options.append(_beam_option)
                    break  # if one witness overruns, they all will, so this beam option is done
            if _new_viable_option_check >= _witness_count or _new_finished_option_check:
                break
            _counter += 1
    _new_options = list(set(_new_options))  # deduplicate
    heapify(_new_options)  # sort from low score to high (low score is best)
    # print(_beam_options)
    if not _new_options and not _finished_options:
        raise Exception("This shouldn't happen: no new options and no finished options")
    else:
        return _new_options[:_beta], _finished_options


def perform_beam_search(_witness_count, _largest_blocks, _block_offsets_by_witness,
                        _witness_offsets_to_blocks, _score_by_block):
    _options, _ = perform_beam_search_step(_witness_count, _largest_blocks, _block_offsets_by_witness,
                                           _witness_offsets_to_blocks, _score_by_block)
    _finished = []  # options that cannot go further
    _counter = 0
    while _options:  # no more options means that we're done
        # TODO: The beam size at the moment is a magic number; can we rationalize it?
        _options, _end_of_life = perform_beam_search_step(_witness_count, _largest_blocks, \
                                                          _block_offsets_by_witness,
                                                          _witness_offsets_to_blocks, _score_by_block,
                                                          _beam_options=_options, _beta=20)
        _finished.extend(_end_of_life)  # add any options that cannot go further
        _counter += 1
    _finished = list(set(_finished))  # TODO: Remove this because we'll sort later?
    _finished.sort(reverse=True, key=lambda f: (sum([_largest_blocks[b][0] for b in f.path]), -1 * len(f.path)))
    return _finished
