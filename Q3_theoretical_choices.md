# Q3 — Theoretical choices

Q3 keeps the mechanism of Q2 (uniform price, welfare objective, BB, soft IR, same reformulation). Only the information changes: the manager posts x before knowing how the members will respond. In practice, the choices we have to make are these.

## 1. Who knows what

- **Structure:** the manager posts x knowing only the distribution of θ; then θ is realized and the members respond. The second stage is not a recourse of the manager but the members' response, so the problem is a stochastic bilevel problem.
- **Uncertain quantity:** the profile if we want to move the volumes (a parameter in the members' constraints), c^L if we want to move the response thresholds (a coefficient in the members' objective)
  -------------------> **We use the profile**
- **IR benchmark:** does the member know its own θ? Yes = ex-post benchmark (computed per scenario). No = ex-ante benchmark (alternative chosen before θ is known).
  -------------------> **We use ex post: min(R, I) per scenario**

## 2. How we represent the uncertainty

- **Scenarios (SAA) or robust / distributionally robust:** the assignment asks for scenarios.
- **N:**
  - small N = optimistic in-sample results, and constraints hold only on the scenarios seen.
  - Large N = the members' KKT systems are replicated N times and the model blows up.
- **Sampling:** nested scenarios or common random numbers, always validated out of sample.
  -------------------> **We use SAA with N ∈ {1, 3, 5, 10, 15, 20}, nested, and 100 out-of-sample scenarios.**

## 3. Attitude towards cost risk (objective)

- **Risk-neutral:** maximize expected welfare.
- **Risk-averse:** expectation plus CVaR, weighted by γ. This cuts the tail at the expense of the mean; here the tail is driven by the penalty α.
- **Worst case:** the limit of CVaR on the sample.
  ------------------->**Base model risk-neutral; one variant with CVaR**

## 4. How we protect the constraints (the core of Q3)

For each constraint (cap, BB, IR) we choose what "satisfied" means:

- **In expectation:** holds on average; any single scenario can violate it.
- **In every scenario:** holds on the whole sample; with one price vector it is often infeasible or expensive.
- **Chance constraint (ε):** controls the frequency of violations, not their size. Joint or individual over hours and members.
- **Bound on the tail of the violation:** controls the magnitude.
- **Hard or soft:** the cap is already soft (penalty); IR is soft because it is infeasible even without uncertainty.

The key question is frequency vs magnitude: the DSO cares *how often* the cap is exceeded, a member *how much* it loses, the manager *how large* the deficit is.
  ------------------->**We use BB and IR in expectation, and a cap chance constraint joint over the hours, with one switch per variant.**

## 5. Solution concept under uncertainty

- The manager stays optimistic on indifferent responses in every scenario.
- Uncertainty makes indifference riskier: a small change in θ moves a member across a threshold and the realized import jumps.
- Out-of-sample evaluation: optimistic response, pessimistic response, or both.
  ------------------->**We evaluate both, and report the range**

## 6. What we measure the value against

- **Deterministic baseline (expected values):** gives the VSS, i.e. what modelling the uncertainty is worth.
- **Perfect information (wait-and-see):** gives the EVPI, i.e. what knowing θ in advance would be worth.
- **Guarantees:** say what each approach guarantees in sample and what it does not guarantee out of sample. For example, a chance constraint at ε on the sample does not guarantee a frequency ≤ ε on new scenarios.
  -------------------> **We report the VSS; the EVPI appears to be optional.**

## Gap in our choices

No variant controls the **magnitude** of the violations directly. Either state it as a limitation in the report, or add an optional variant (e.g. CVaR on the IR shortfall or on the budget deficit)
I talk about this in a point in the Q3_assumptions.md file
