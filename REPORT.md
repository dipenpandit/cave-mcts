# Scale-Free, Variance-Aware Exploration for Heuristic-Guided Monte Carlo Tree Search in Baaghchaal

## Abstract

Early Playout Termination (EPT) makes Monte Carlo Tree Search (MCTS) more
sample-efficient by replacing deep random playouts with a heuristic evaluation.
It also introduces a scale problem: the heuristic may be unbounded while the
UCT exploration bonus remains $O(1)$. We study this problem in Baaghchaal, an
asymmetric game with tiger and goat roles, and introduce CAVE (Calibrated,
Asymmetric, Variance-aware Exploration). CAVE calibrates evaluations to
role-specific values in $[0,1]$, keeps heuristic and sampled values in separate
channels, and uses a Bernstein-style confidence radius with a role-specific
constant. A root diagnostic gives an exploration-to-exploitation spread ratio
of $0.011$ for the raw heuristic rule and ratios close to one after bounding the
value scale. These results establish the scale mismatch; comparative playing strength experiments remain outside the current result set.

## 1. Introduction

UCT assumes that rewards have a known bounded scale. The heuristic-guided rule
used as the prior baseline is

$$\operatorname{UCT}_{\mathrm{enh}}(j) = \bar{E}_j + C_p\sqrt{\frac{2\ln n}{n_j}}$$

where $\bar{E}_j$ is the mean raw evaluation of child $j$. In Baaghchaal,
$\bar{E}_j$ can span hundreds of points, whereas the exploration term is usually
small. The resulting selection rule is effectively greedy, and the apparent
insensitivity to $C_p$ is a scale artefact rather than evidence of robustness.
In the notation above, $n$ is the number of visits to the parent and $n_j$ is
the number of visits to child $j$.

EPT stops a simulation after a fixed depth and evaluates the resulting state
instead of continuing to a terminal position. This reduces the cost of a
simulation, but it also makes the scale of the evaluation function part of the
selection rule. The present study isolates that interaction rather than
changing the underlying game evaluation.

The problem is amplified by asymmetry. Tigers and goats have different
branching factors, objectives, and evaluation statistics. A single exploration
constant is therefore unlikely to be appropriate for both roles.

This work makes three contributions:

1. It measures the relative scale of exploitation and exploration at the root.
2. It defines CAVE, a bounded, dual-channel, variance-aware selection rule.
3. It provides a common MCTS driver and experiment suite for role-separated
   comparisons.

## 2. Method

### 2.1 Role-specific calibration

Let $E_r(s)$ denote the raw evaluation of state $s$ for role $r$. We use a
logistic mapping fitted separately for tigers and goats:

$$
q_r(s) = \sigma\bigl(\beta_r(E_r(s)-\mu_r)\bigr),
\qquad \sigma(z)=\frac{1}{1+e^{-z}}.
$$

The fitted value is bounded in $[0,1]$, which puts exploitation and exploration
on a common scale. Values near $1$ indicate that the state is favorable to the
role being evaluated, while values near $0$ indicate an unfavorable state.
Calibration is applied when an EPT playout reaches its depth limit; terminal
positions still use their known win, loss, or draw outcome. An online z-score
calibrator is available when no fitted parameters are supplied.

The parameters are role-specific because the same raw score does not have the
same meaning for tigers and goats. Fitting $\beta_r$ controls how quickly the
value changes as the evaluation changes, while $\mu_r$ sets the raw score that
maps to the middle of the bounded range.

### 2.2 Dual-channel exploitation

CAVE stores two estimates for each child. The sampled mean $\bar{Q}_j$ is the
average of the values returned by simulations and backed up through that child.
The implicit-minimax value $V_j$ is a separate heuristic estimate propagated
through the tree by taking the best available child value from the relevant
player's perspective. Their blended exploitation value is

$$
\hat{Q}_j = (1-\alpha)\bar{Q}_j + \alpha V_j.
$$

The parameter $\alpha$ controls the influence of the heuristic channel. This
allows tactical information from the heuristic to influence selection without
replacing or altering the statistical estimate collected from simulations.
Whenever a value is compared at a parent node, it is expressed in that
parent's role perspective; this is why backed-up values are flipped between
the tiger and goat frames.

### 2.3 Variance-aware exploration

Let $\hat{V}_j$ be the empirical variance of values backed up through child $j$,
computed from both the running sum and the running sum of squares. CAVE uses the
role-specific bonus

$$
U_j = C_r\left[
\sqrt{\frac{2\hat{V}_j\ln n}{n_j}} + \frac{3\ln n}{n_j}
\right].
$$

The complete selection score is $\hat{Q}_j+U_j$. At each selection step, MCTS
chooses the child with the largest score, so the bonus determines how much a
child's uncertainty can compensate for a lower current value estimate. A
low-variance child loses its bonus quickly, while an unresolved branch retains
exploration budget. The final term also keeps rarely visited children from
receiving zero bonus. The constant $C_r$ may differ between roles because the
roles have different branching factors and search behavior.

## 3. Experimental design

All policies use the same MCTS driver and differ only in their policy object.
Comparisons match the number of simulations per move rather than wall-clock
time. Tiger and goat results are reported separately, draws receive score
$0.5$, and Wilson intervals are used for match scores.

The repository provides four studies:

- `run_diagnostics.py` measures the root scale mismatch.
- `run_calibration.py` fits the role-specific logistic mappings.
- `run_ablation.py` compares the baseline and successive CAVE components.
- `run_sensitivity.py` sweeps the exploration constant for the baseline and
  CAVE.

The diagnostic prints its measurements, while the other three studies write
JSON files under `results/`. Ablation and sensitivity scores are role-specific:
a result for the tiger side is not pooled with a result for the goat side.

## 4. Results

The diagnostic statistic is

$$
\rho = \frac{\max_j U_j-\min_j U_j}
{\max_j \hat{Q}_j-\min_j \hat{Q}_j}.
$$

Across six mid-game positions with 300 simulations per position, the observed
spreads were:

| Selection rule | Exploitation spread | Exploration spread | $\rho$ |
|---|---:|---:|---:|
| Raw-heuristic UCT | 332.58 | 3.79 | **0.011** |
| Min-max normalisation | 0.10 | 0.10 | 1.02 |
| Logistic calibration | 0.14 | 0.14 | 1.03 |
| CAVE | 0.20 | 0.20 | 1.04 |

The raw rule assigns approximately one percent of the root decision spread to
exploration. The bounded variants bring the two terms to comparable scales.
This is evidence for the scale-mismatch diagnosis, not evidence that CAVE
already improves playing strength. In general, $\rho\ll1$ indicates that
exploration is nearly inert, while $\rho\approx1$ indicates that exploration
has variation comparable to exploitation.

Preliminary self-play calibration produced:

$$
\beta_{\mathrm{tiger}}\approx0.0142,\quad
\mu_{\mathrm{tiger}}\approx277,\quad
\beta_{\mathrm{goat}}\approx0.0070,\quad
\mu_{\mathrm{goat}}\approx-147.
$$

The difference supports fitting separate mappings for the two roles. Completed
ablation and sensitivity match results are not included in this version.

## 5. Limitations and conclusion

Baaghchaal is a draw under optimal play, so stronger agents may produce many
draws and require larger samples. Calibration inherits the distribution of the
self-play positions used for fitting. The handcrafted evaluation is held fixed;
the contribution is a change to selection, not a new evaluation function.

The current evidence shows that raw heuristic scale can make UCT exploration
numerically ineffective and that calibration restores a commensurate value
range. The remaining question is whether this restored control produces a
reliable playing-strength improvement for both roles, which requires the
completed match studies.

## References

- Auer, P., Cesa-Bianchi, N., and Fischer, P. (2002). Finite-time analysis of
  the multiarmed bandit problem.
- Kocsis, L. and Szepesvári, C. (2006). Bandit based Monte-Carlo planning.
- Audibert, J.-Y., Munos, R., and Szepesvári, C. (2009). Exploration--
  exploitation tradeoff using variance estimates in multi-armed bandits.
- Lanctot, M., Winands, M., Pepels, T., and Sturtevant, N. (2014). Monte Carlo
  tree search with heuristic evaluations using implicit minimax backups.
- Lorentz, R. (2015). Early playout termination in MCTS.
- Schrittwieser, J. et al. (2020). Mastering Atari, Go, chess and shogi by
  planning with a learned model.
- Lim, Y. J. and Nievergelt, J. (2004). Computing Tigers and Goats.
