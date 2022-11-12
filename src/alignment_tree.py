from collections import deque
from dataclasses import dataclass


@dataclass
class Node:
    """Nodes of all types have integer id properties"""
    id: int


@dataclass
class Branching_node(Node):
    """Branching nodes have children, either a list of nodes
    or a list of two-item tuples, one per witness, for slicing
    into the original token array
    """
    children: list
    processed: bool
    absolute_offsets: list


@dataclass
class Leaf_node(Node):
    """Leaf nodes may be aligned or non-aligned and have string values"""
    string: str
    aligned: bool


@dataclass
class Alignment_tree:
    root: Node
    node_by_id: dict


def traverse_tree(r):
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
