Pseudo-Random Multinomial Distribution
======

A package for drawing samples from pseudo-random multinomial distribution.

## Background

This is originally meant to be used in our unpublished work, where we only concern about pseudo-random binomial
distribution.
In essence, we want to generate 0/1 bits such that 0 or 1 events are promoted to occur consecutively, forming clusters,
while we can choose expected number of consecutive occurrences and overall nominal probability.

The code in this repo is further extended to pseudo-random multinomial distribution and can support even more behaviors.
Therefore, we will add some background here.

### Main Idea

### Notations

|    Term    |                                           Meaning                                           |
|:----------:|:-------------------------------------------------------------------------------------------:|
|    $S$     |                                   Chain transition matrix                                   |
|   $E_i$    |               Expectation (number of consecutive occurrences) of an event $i$               |
|   $P_i$    |                              Nominal probability of event $i$                               |
|   $\pi$    |         Stationary distribution, $\pi^i_j$ is stationary of state $j$ of chain $i$          |
| $\alpha_j$ | _Linger_ probability $\alpha_j = 1 - e_j$, where $e_j$ is the exit probability of state $j$ |

### Expectation

The expectation of an elementary chain is computed as sum of cumulative product of lingering probability $a_k$ (the
probability of changing to the next state instead of exiting the chain).
This can be easily proved using different methods.

|             |                          Finite chain                          |                           Infinite chain                            |
|:-----------:|:--------------------------------------------------------------:|:-------------------------------------------------------------------:|
| Expectation | $E=1+\sum\limits_{j=1}^{n}{\prod\limits_{k=1}^{j}{\alpha_k}}$  | $E=1+\sum\limits_{j=1}^{\infty}{\prod\limits_{k=1}^{j}{\alpha_k}}$  |

The expectation of finite chains (with reasonably small number of states) can be computed easily.
Meanwhile, for infinite chains, the sum of cumulative product of linger probabilities is a series that always converge
if $\lim_{k\rightarrow\infty}{\alpha_k}=0$ (or $\lim_{k\rightarrow\infty}{e_k}=1$).
It's limit can be estimated using Wynn's Epsilon method.

Then, the expectation of each chain $\text{\bold{E}}_i$ in a system multiple chains is computed as:
```math
\text{\bold{E}}_i=\frac{E_i}{1-S_{ii}}
```

### Nominal Probability

To compute nominal probabilities, we need first to compute the entrance stationary $\pi^i_1$ of each chain.
This is done by solving the following system of linear equations:

```math
\left\lbrace\begin{array}{@{}l@{}l@{}l@{}l@{}l@{}}
S_{11}\pi^1_1 &+ S_{21}\pi^2_1 &+ \cdots &+ S_{N1}\pi^N_1 &= \pi^1_1 \\ 
S_{12}\pi^1_1 &+ S_{22}\pi^2_1 &+ \cdots &+ S_{N2}\pi^N_1 &= \pi^2_1 \\ 
 & & \vdots & & \\ 
S_{1N}\pi^1_1 &+ S_{2N}\pi^2_1 &+ \cdots &+ S_{NN}\pi^N_1 &= \pi^N_1 \\ 
E_1\pi^1_1 &+ E_2\pi^2_1 &+ \cdots &+ E_N\pi^N_1 &= 1 \quad\quad (\sum\limits_{ij}{\pi^i_j}=1)
\end{array}\right.
```

Note that the last equation contains the relationship between the expectation, entrance stationary, and probability 
of event $i$:

$$ P_i = E_1\pi^1_1 $$

## Requirements

- Python 3.6+
- numpy
- cython (to compile)
- cy-root

## Installation

`pseudo-multinomial` has pre-built binaries hosted at TestPyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ pseudo-multinomial
```

Pull this repo and install from source:
```bash
pip install .
```

## Usage

#### Example 1:
In this example, we replicate the perfect coin tossing:
```python
from pseudo_multinomial import MasterChain, ForwardingChain

g = MasterChain(
    chains=[ForwardingChain(),  # this chain has 100% exit probability
            ForwardingChain()],
    chain_transition_matrix=[
        [.5, .5],  # both chains have 50% chance to
        [.5, .5],  # be chosen.
    ])

print(g.pseudo_multinomial(100))
```
Output:
```
[1 0 1 0 0 1 1 0 1 1 0 1 0 0 0 0 0 0 0 1 1 1 0 0 0 0 1 1 0 1 1 1 0 0 1 0 1 0 0 1 0 0 1 0 1 0 0 0 1 1 0 1 1 0 1 0 0 0 1 0 0 0 0 1 1 0 0 1 0 0 1 1 1 1 0 0 1 0 1 0 1 1 1 0 0 1 0 1 0 0 1 1 1 0 1 0 0 0 0 1]
```

By modifying the chain transition matrix $S$, we can create different behaviors:
```python
g = MasterChain(
    chains=[ForwardingChain(),
            ForwardingChain()],
    chain_transition_matrix=[
        [0, 1],  # alternate between chains
        [1, 0],
    ])

print(g.pseudo_multinomial(100))
```
Output:
```
[1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0]
```

## License

The code is released under the MIT license. See [`LICENSE.txt`](LICENSE.txt) for details.
