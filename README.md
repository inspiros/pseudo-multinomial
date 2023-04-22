Pseudo-Random Multinomial Distribution
======

A package for drawing samples from pseudo-random multinomial distribution.

## Background

This is originally meant to be used in our unpublished work, where we only concern about pseudo-random binomial
distribution.
In essence, we want to generate 0/1 bits such that 0 or 1 events are promoted to occur consecutively, forming clusters,
while we can choose expected number of consecutive occurrences and overall nominal probability at will.

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

The base expectation of an elementary chain is computed as sum of cumulative product of lingering probability $a_k$ (the
probability of changing to the next state instead of exiting the chain).
This can be easily proved using different methods.

|                          |                          Finite chain                          |                           Infinite chain                            |
|:------------------------:|:--------------------------------------------------------------:|:-------------------------------------------------------------------:|
| Base Expectation Formula | $E=1+\sum\limits_{j=1}^{n}{\prod\limits_{k=1}^{j}{\alpha_k}}$  | $E=1+\sum\limits_{j=1}^{\infty}{\prod\limits_{k=1}^{j}{\alpha_k}}$  |

The formula of finite chains (with reasonably small number of states) can be easily computed.
Meanwhile, for infinite chains, the sum of cumulative product of linger probabilities is a series that always converge
if $\lim_{k\rightarrow\infty}{\alpha_k}=0$ (or $\lim_{k\rightarrow\infty}{e_k}=1$).
It's limit can be estimated using [Wynn's Epsilon method](https://mathworld.wolfram.com/WynnsEpsilonMethod.html).

Then, the (real) expectation of each chain $\mathbf{E}_i$ in a system multiple chains is computed as:
```math
\mathbf{E}_i=\frac{E_i}{1-S_{ii}}
```

### Probability

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
In this example, we replicate the perfect coin flip using a pair of `ForwardingChain`:
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

#### Example 2:
In this example, we create an unnatural 6-faced dice that usually generate consecutive results using
the `HarmonicChain`:
```python
from pseudo_multinomial import MasterChain, HarmonicChain

g = MasterChain.from_pvals(
    chains=[HarmonicChain(c=.5),
            HarmonicChain(c=.5),
            HarmonicChain(c=.5),
            HarmonicChain(c=.5),
            HarmonicChain(c=.5),
            HarmonicChain(c=.5)],
    pvals=[1/6] * 6,
    repeat=True)

print(g.pseudo_multinomial(100))
```
Output:
```
[2 0 0 0 0 3 3 3 1 1 5 5 0 0 0 1 1 1 0 2 2 2 5 1 1 1 1 1 5 4 1 5 5 0 0 0 0 4 4 4 4 4 3 0 5 5 5 5 5 0 0 4 1 4 4 0 0 2 3 3 3 2 2 4 4 4 4 4 4 3 4 4 4 4 4 4 4 4 3 1 1 5 5 5 4 2 5 4 4 4 5 5 1 2 1 1 1 0 0 0]
```
However, the nominal probability of each event is still exactly $1/6$:
```python
print(g.probs())
# [0.16666667 0.16666667 0.16666667 0.16666667 0.16666667 0.16666667]
```

#### Example 3: The Dota 2 Pseudo-random _Binomial_ Distribution
If you are a **Warcraft 3** or **Dota 2** fan, you may notice that these games have a statistical mechanic affecting
certain skills or items.
It discourages the consecutive occurrence of success events to limit the influence of luck.
It works by slowly increasing the success probability by a constant $c$ after every failure event, and reset to
$c$ after a success event.

We can recreate this with a `LinearChain` with `initial_state=2` for failure event (0) and a `ForwardingChain` for
success event (1).
The success chain can also repeat itself with probability $S_{22}=c$, while the failure chain always transit to success
chain after exiting $S_{12}=1$.

This example also shows the use of the `RootFindingSolver` to optimize a single parameter $c$ to get a desired nominal
probability of success event:
```python
from pseudo_multinomial import MasterChain, LinearChain, ForwardingChain
from pseudo_multinomial.solver import RootFindingSolver

c = 0.3  # approximate p=0.5
g = MasterChain(
    chains=[LinearChain(c=c, initial_state=2),
            ForwardingChain()],
    chain_transition_matrix=[
        [0., 1.],
        [1 - c, c],
    ])


def update_param_fn(c):  # function to update parameter c
    g.chains[0].c = c
    g.S[1, 1] = c
    g.S[1, 0] = 1 - c


p = 0.05
while p < 1:
    solver = RootFindingSolver(objective_fn=lambda: g.probs()[1],
                               objective_val=p,
                               update_param_fn=update_param_fn)
    solver.solve(method='bisect', a=1e-7, b=1, etol=1e-15, ertol=0, ptol=0, prtol=0)
    print(f'desired_p={p:.02f}, solved_p={g.probs()[1]:.05f}, c={g.chains[0].c}')
    p += .05
```
Output:
```
desired_p=0.05, solved_p=0.05000, c=0.003801658303553148
desired_p=0.10, solved_p=0.10000, c=0.01474584478107244
desired_p=0.15, solved_p=0.15000, c=0.03222091437308729
desired_p=0.20, solved_p=0.20000, c=0.055704042949781624
desired_p=0.25, solved_p=0.25000, c=0.08474409185231699
desired_p=0.30, solved_p=0.30000, c=0.11894919272540436
desired_p=0.35, solved_p=0.35000, c=0.15798309812574696
desired_p=0.40, solved_p=0.40000, c=0.20154741360775413
desired_p=0.45, solved_p=0.45000, c=0.24930699844016374
desired_p=0.50, solved_p=0.50000, c=0.3021030253487422
desired_p=0.55, solved_p=0.55000, c=0.36039785093316745
desired_p=0.60, solved_p=0.60000, c=0.4226497308103737
desired_p=0.65, solved_p=0.65000, c=0.4811254783372293
desired_p=0.70, solved_p=0.70000, c=0.5714285714285714
desired_p=0.75, solved_p=0.75000, c=0.666666666666665
desired_p=0.80, solved_p=0.80000, c=0.7500000000000004
desired_p=0.85, solved_p=0.85000, c=0.8235294117647065
desired_p=0.90, solved_p=0.90000, c=0.8888888888888895
desired_p=0.95, solved_p=0.95000, c=0.9473684210526316
```

See more: https://dota2.fandom.com/wiki/Random_Distribution

## License

The code is released under the MIT license. See [`LICENSE.txt`](LICENSE.txt) for details.
