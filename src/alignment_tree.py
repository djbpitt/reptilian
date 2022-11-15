"""Tree of nodes

TODO: Replace our own classes with networkx
Nodes have a type property: aligned, unaligned, branching
Each of the three types requires specific properties and prohibits others
TODO: Can we formalize those requirements and prohibitions? Can a networkx
    property be a complex object?
"""

from collections import deque
from dataclasses import dataclass
from typing import List


@dataclass
class Node:
    """Superclass for nodes of all types

    Properties:
        id: unique integer identifier, set with len(tree.node_by_id)
        token_ranges: list of tuples, one per witness, with start and
            exclusive end offsets in global token list
    """
    id: int
    token_ranges: List[tuple]


@dataclass
class Branching_node(Node):
    """Branching nodes have children only after expansion"""
    expanded: bool
    children: List[Node]


@dataclass
class Leaf_node(Node):
    """Leaf nodes may be aligned or non-aligned

    Unaligned leaf nodes have empty strings (we’ll get the tokens later)
    Aligned leaf nodes have token strings as a development convenience
    TODO: Except as a development convenience, should we add token
        strings only during visualization?"""
    string: str # empty if not aligned
    aligned: bool


@dataclass
class Alignment_tree:
    """Root is a branching node with an associated node_by_id dictionary

    Use len(root.node_by_id) to create id for new node
    TODO: Decide whether root.node_by_id should be public"""
    root: Branching_node
    node_by_id: dict


def traverse_tree(r:Alignment_tree) -> List[Node]:
    """Depth-first traversal, return list of all nodes
    TODO: Replace list with generator?

    Push root (input) onto stack (just once)
    Loop: Pop top of stack, process, push all children onto stack
        NB: If children are A, B, C, A should be leftmost after being added
    Exit when stack is empty
    """
    _l = []  # list of nodes to return
    _s = deque()  # extendleft() and popleft() to manipulate
    _s.appendleft(r)  # root is special case
    while _s:
        current_node = _s.popleft()
        _l.append(current_node)
        if isinstance(current_node, Branching_node):
            # sequence of children should be added with leftmost at left edge
            # extendleft() reverses, so we reverse ourselves to retain original order
            _s.extendleft(current_node.children[::-1])
    return _l


def expand_node(_tree: Branching_node, _parent:Branching_node):
    """Expand children of previously unexpanded branching node

    Parameters:
        _tree: Tree has root and node_by_id dictionary
        _parent: node to expand

    Returns:
          Modified tree
    """
    pass