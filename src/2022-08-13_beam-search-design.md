# Beam search design

2022-08-13

## Overview

* Eventual beam result contains fixed number of beam options (*β*)
* Search proceeds hierarchically; each step (tier) contains that number of beam options
* At each tier, evaluate all possible continuations of each beam option
* After evaluation of entire tier, keep *β* best beam options according to score (see below)
* **Score** is number of tokens aligned, where low is better because the end goal is to align as many as possible, and a low interim score has the most potential.

## Complexity

If *β* = 3 and the witness count = 5, we perform 15 (+ 3; see immediately below) evaluations on each tier and keep and advance the 3 best to the next tier. The complexity equals the size of the beam (*β*) times the number of evaluations to perform for each beam option (in our case, the number of witnesses (5) plus 1 for the “skip” alternative).

Witnesses will often agree, and if we track the ones that are covered by a particular beam option we can avoid performing essentially the same computation for each of the agreeing witnesses. E.g., if A and B agree, when we compute A we can record that it also covers B, and we can then short-circuit around B when it comes up in the queue.

## Lessons learned

1. Passing the full data set through a cascade of functions is a code smell. A function chain should normally reduce the number of parameters at each step, so each inner function doesn’t have to know about surrounding data.
2. **Exit condition:** Each beam option becomes a list of blocks taken, so that the end result is a list of blocks that contains the majority order. When we reach the final token of the majority of witnesses, we’re done.
3. We need to reassess how we evaluate the options. Start from shared Start token, advance one position in each witness (plus skip), and compute the score for advancing all pointers for all witnesses in sync with that one. The maximum number of scorings equals the number of witnesses plus one.
4. **Caution:** Is it possible that *all* first blocks are transposed? (No, because the Start token must be part of the eventual graph.)
