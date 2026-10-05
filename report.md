# CAVE-MCTS Research Report

## Summary

This report evaluates a scale mismatch in heuristic-guided Monte Carlo Tree
Search (MCTS) for Baaghchaal. The prior selection rule combines an unbounded
heuristic evaluation with a bounded UCT exploration term:

$$
\operatorname{UCT}_{\mathrm{enh}}(j) = \bar{E}_j + C_p\sqrt{\frac{2\ln n}{n_j}}.
$$

Because $\bar{E}_j$ can be much larger than the exploration bonus, changing
$C_p$ has little effect on child selection. The search consequently behaves
closer to greedy heuristic descent than to a balanced exploration--exploitation
method.

## Proposed Method

CAVE (Calibrated, Asymmetric, Variance-aware Exploration) addresses this
mismatch in three stages. First, a role-specific logistic calibration maps raw
evaluations to the interval $[0,1]$:

$$
q_r(s) = \sigma\bigl(\beta_r(E_r(s)-\mu_r)\bigr),
\qquad \sigma(z)=\frac{1}{1+e^{-z}}.
$$

Second, the sampled mean $\bar{Q}_j$ and the heuristic minimax value $V_j$ are
kept as separate channels and combined as

$$
\hat{Q}_j = (1-\alpha)\bar{Q}_j + \alpha V_j.
$$

Third, CAVE uses a role-specific Bernstein-style exploration radius:

$$
U_j = C_r\left[
\sqrt{\frac{2\hat{V}_j\ln n}{n_j}} + \frac{3\ln n}{n_j}
\right],
$$

where $\hat{V}_j$ is the empirical variance of backed-up values. The complete
selection score is therefore

$$
\operatorname{CAVE}(j) = \hat{Q}_j + U_j.
$$

The role-specific constant $C_r$ reflects the different branching factors and
planning horizons of tigers and goats.

## Diagnostic Result

The diagnostic measures the relative influence of exploration at the root:

$$
\rho = \frac{\max_j U_j-\min_j U_j}
{\max_j \hat{Q}_j-\min_j \hat{Q}_j}.
$$

Across six mid-game positions with 300 simulations per position, the raw
heuristic rule produced an exploitation spread of $332.58$ and an exploration
spread of $3.79$, giving $\rho=0.011$. The exploration term therefore
contributed approximately one percent of the decision spread.

| Selection rule | Exploitation spread | Exploration spread | $\rho$ |
|---|---:|---:|---:|
| Raw-heuristic UCT | 332.58 | 3.79 | **0.011** |
| Min--max normalisation | 0.10 | 0.10 | 1.02 |
| Logistic calibration | 0.14 | 0.14 | 1.03 |
| CAVE | 0.20 | 0.20 | 1.04 |

The bounded variants restore the exploration term to the same numerical scale
as the exploitation term. This result supports the scale-mismatch diagnosis;
it does not by itself establish a playing-strength improvement.

## Calibration Result

Preliminary self-play fitting produced distinct role-specific parameters:
$\beta_{\mathrm{tiger}}\approx0.0142$,
$\mu_{\mathrm{tiger}}\approx277$,
$\beta_{\mathrm{goat}}\approx0.0070$, and
$\mu_{\mathrm{goat}}\approx-147$. The difference supports using separate
calibration models for the two roles.

## Experimental Scope

The implementation includes drivers for calibration, ablation, diagnostics, and
sensitivity analysis. The present repository contains the diagnostic table and
preliminary calibration values, but it does not include completed ablation or
sensitivity match results. Accordingly, no comparative win-rate claim is made
here. The intended evaluation compares all policies at matched simulation
budgets, reports tiger and goat results separately, and uses Wilson intervals
with draws scored as $0.5$.

## Limitations

Baaghchaal is a draw under optimal play, so stronger agents may produce more
draws and require larger samples for reliable comparisons. The handcrafted
evaluation functions are held fixed; the contribution concerns the selection
rule rather than the quality of the evaluation itself. Calibration estimates
also inherit the distribution and biases of the self-play positions used to fit
them.
