from pseudo_multinomial import MasterChain, LinearChain, ForwardingChain
from pseudo_multinomial.solver import RootFindingSolver


def main():
    c = 0.422649730810374235490851220
    g = MasterChain([
        LinearChain(c=c, initial_state=2),
        ForwardingChain(),
    ], chain_transition_matrix=[
        [0., 1.],
        [1 - c, c],
    ])

    def update_param_fn(c):
        g.chains[0].c = c
        g.S[1, 1] = c
        g.S[1, 0] = 1 - c

    p = 0.05
    while p < 1.01:
        solver = RootFindingSolver(objective_fn=lambda: g.probs()[1],
                                   objective_val=p,
                                   update_param_fn=update_param_fn)
        solver.solve(method='bisect', a=1e-7, b=1, etol=1e-15, ertol=0, ptol=0, prtol=0)
        print(f'desired_p={p:.02f}, solved_p={g.probs()[1]:.05f}, c={g.chains[0].c}')
        p += .05


if __name__ == '__main__':
    main()
