# CAVE-MCTS

CAVE-MCTS is a follow-up research follow-up research implementation of heuristic-guided Monte Carlo Tree
Search for Baaghchaal. The prior selection rule combines an unbounded
heuristic evaluation with a bounded UCT exploration term:

$$
\text{UCT}_{\mathrm{enh}}(j) = \bar{E}_j + C_p\sqrt{\frac{2\ln n}{n_j}}.
$$

Because $\bar{E}_j$ can be much larger than the exploration bonus, changing
$C_p$ has little effect on child selection. The search consequently behaves
closer to greedy heuristic descent than to a balanced exploration-exploitation
method.

It studies how the scale of a handcrafted evaluation
affects the exploration term in UCT and implements a calibrated,
variance-aware alternative.

## Installation

The project requires Python 3.14 or later.

```bash
uv venv --python 3.14
source .venv/bin/activate
uv sync
```

## Experiments

Run an experiment from the repository root with:

```bash
uv run python -m cave-mcts.experiments.run_diagnostics
uv run python -m cave-mcts.experiments.run_calibration --games 60
uv run python -m cave-mcts.experiments.run_ablation --games 20 --sims 400
uv run python -m cave-mcts.experiments.run_sensitivity --games 12 --sims 300
```

The calibration, ablation, and sensitivity studies write JSON output to
`results/`. The diagnostic study prints its summary directly.

The studies have different purposes: calibration estimates the role-specific
logistic parameters, diagnostics measures the relative scale of the two
selection terms, ablation compares individual CAVE components, and sensitivity
shows whether changing the exploration constant changes agent performance.

## Method

The prior heuristic-guided rule selects child $j$ using

$$
\text{UCT}_{\mathrm{enh}}(j) = \bar{E}_j + C_p\sqrt{\frac{2\ln n}{n_j}},
$$

where $\bar{E}_j$ is an unnormalised heuristic value. Since the heuristic can
be much larger than the $O(1)$ exploration term, $C_p$ can have little effect
on selection. Here $n$ is the parent visit count and $n_j$ is the visit count
for child $j$.

CAVE maps each role-specific heuristic to $[0,1]$, keeps statistical and
heuristic values separate, and adds a variance-aware confidence term:

$$
q_r(s) = \sigma\bigl(\beta_r(E_r(s)-\mu_r)\bigr),
\qquad \sigma(z)=\frac{1}{1+e^{-z}},
$$

$$
\text{CAVE}(j) = (1-\alpha)\bar{Q}_j + \alpha V_j + C_r\left[
\sqrt{\frac{2\hat{V}_j\ln n}{n_j}} + \frac{3\ln n}{n_j}\right].
$$

Here $V_j$ is an implicit-minimax heuristic value, $\hat{V}_j$ is the
empirical variance of backed-up values, and $C_r$ is selected separately for
tigers and goats.

## Package layout

- `game.py`: Baaghchaal state representation, legal moves, captures, and
  terminal conditions.
- `heuristics.py`: raw evaluations and logistic or online calibration.
- `policies.py`: baseline, normalised, calibrated, and CAVE selection rules.
- `mcts.py`: the shared MCTS driver and random agent.
- `arena.py`: matches, scoring, and Wilson intervals.
- `experiments/`: calibration, diagnostic, ablation, and sensitivity studies.
- `paper/`: Markdown and LaTeX research documents.

All compared policies use the same search driver. The policy object is the only
part that changes between ablation variants.

## Current evidence

The root diagnostic uses

$$
\rho = \frac{\text{spread}(\text{bonus})}
{\text{spread}(\text{exploitation})}.
$$

For six mid-game positions and 300 simulations per position, the raw heuristic
rule produced an exploitation spread of $332.58$, an exploration spread of
$3.79$, and $\rho=0.011$. Min--max normalisation, calibration, and CAVE produced
ratios near one. This supports the scale-mismatch diagnosis, but it is not a
playing-strength result. Values of $\rho$ much smaller than one mean that
exploration has little influence on the selected child; values near one mean
that both terms have comparable variation.

Preliminary calibration estimates are role-specific:
$\beta_{\mathrm{tiger}}\approx0.0142$,
$\mu_{\mathrm{tiger}}\approx277$,
$\beta_{\mathrm{goat}}\approx0.0070$, and
$\mu_{\mathrm{goat}}\approx-147$.

The implementation includes match drivers for further ablation and sensitivity
measurements. Their comparative win-rate results are not included in the
current repository snapshot.

## Rules

The implementation follows the Lim-Nievergelt rule set: goats place all twenty
pieces before sliding, tigers capture by jumping, tigers win after five
captures, and goats win when no tiger has a legal move. Three-fold repetition
and fifty plies without progress are treated as draws.
