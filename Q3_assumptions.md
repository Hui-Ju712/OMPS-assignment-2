# Question 3 — Assumptions

Tags such as "FA question 1.3" refer to the point of the **Formative assessment** that supports the choice

## A. Choices inherited from Questions 1 and 2

- **An imported kWh above the cap pays the day-ahead price plus the penalty, instead of the discounted tariff. (FA question 1.1)**
  
- **The daily energy of each load is imposed as an equality, equal to the daily reference energy. (FA question 2 – A2)**
  - should we add a sentence in the report, like "the equality is equivalent to the minimum of the assignment, because every effective price is positive, so no member ever consumes more than required"? I think there isn't one yet...

- **The manager posts one uniform price per hour, identical for all members; members are charged on their net consumption. (FA question 2 – C1)**
  - Required by the assignment; no justification needed.
  - Per-member prices are an extension of Q2 only, not the basis of Q3. keep in mind while drafting the report for Q3. Is a detail but I feel like is important

- **The manager is optimistic: when a member is indifferent, it picks the response the manager prefers. (FA question 1.5)**
  - Required by the assignment; no justification needed.
  - In Q3, keep the optimistic concept in-sample, but evaluate out-of-sample with both the optimistic and the pessimistic response, to show the manager's exposure.

- **The complementarity conditions are linearized with SOS1 constraints, not big-M. (FA question 1.6)**
  - If my understanding is correct, we should update section 2, the appendix and the nomenclature of the draft, which still describe big-M constants and binaries.

- **The members' payments are linearized through strong duality. (FA question 1.6)**
  - we should explain it once in Q2 and in Q3 only state that it is applied per scenario.
  - a very similar transformation is detailed in the Week 7 lecture

- **The meter split is enforced exactly with SOS1 constraints, so the budget cannot be balanced with fictitious penalties or tariffs. (FA question 1.7)**
  - The feedback recommends a lighter approach: no extra constraints, a check after solving, and binaries only if the check fails.
  Our choice is valid but adds 48 SOS1 sets per scenario in Q3. Still, I think is not an issue.
  We should explain it once in Q2 though

- **The posted prices are bounded between the discounted export price and the import price above the cap. (FA question 1.4, FA question 2 – C1)**
  - I see that the bounds of −5 and 15 DKK/kWh in the appendix table of the draft, but the teacher says that putting a lower bound negative is incorrect in a logic way, is like paying the owner of the battery for charging and dicharging at the same time. We should change that and put it to 0 I think

- **Among welfare-optimal prices, the manager picks the prices closest to the marginal cost at the meter (quadratic distance), applied lexicographically after the welfare. (FA question 1.4)**
This is something I did not understand but Lesia prefers it, in particular this "lexicographic" approachm, so I just put a small list of todo that could help the coding team potentially:
  - Justify in one sentence: it selects among optimal prices without changing the feasible set and without tuning a weight.
  - State that it changes how the rent is shared, and show its effect on the distribution.
  - Not implemented yet: add it to the code.
  - Each solve becomes two solves (welfare, then regularization); account for it in the solution times.

- **Budget balance is an equality and a hard constraint. (FA question 1.3, FA question 2 – C3)**
  - Required by the assignment; no justification needed.
  - In Q3 it is imposed in expectation

- **Individual rationality is soft: one slack per member, penalized in the objective with a large weight. (FA question 1.3)**
  - Justify it: hard IR is infeasible in the base case with a uniform price.
  - The feedback prefers the lexicographic version (minimize the slacks first, then maximize welfare); a large penalty is accepted as an alternative.
  - The members with a positive slack are the blocking members; the sum of the slacks is what would have to be given up. Mention that the blocking set may not be unique.
  - The weight must be verified: the sum of slacks must equal the lexicographic minimum and must not change when the weight is doubled. Run this check on DP and on the stochastic model with 5 scenarios.

- **The IR benchmark is the better of no contract and the individual contract, for each member. (FA question 1.2)**
  - Justify in one sentence: it is the strictest test, since a member that passes it prefers the community to any stand-alone option.
  - we should use one symbol for the benchmark in Q2 and Q3, instead of using two different symbols

- **The contract terms are the base case. (FA question 3 – Protocol)**
  - No justification needed.
  - Use a separate symbol for the cap scaling of the sweeps: the draft uses the same letter for the assignment's factor (1.3) and for the scaling around the base case (1). ( I already put a note about this in the overleaf)
  - In Q3 the cap and the individual shares are fixed "ex ante" and are not recomputed per scenario.

## B. New choices in Question 3

- **Only the members' reference profiles are uncertain; the disutility slope is not.**
  - Consistent with the base study of the assignment; no justification needed.
  - Uncertainty on the slope remains an optional variation, only if solution times allow.

- **The uncertainty hyper-parameters are the base values of the data file (uniform scaling between 0.8 and 1.2, equiprobable scenarios).**
  - No justification needed.
  - A concentration sweep (2 and 10) is optional; it cannot replace the mandatory sensitivity study.

- **Each member knows its own preferences; only the manager is uncertain.**
  - Justify in one sentence, quoting the problem statement.
  - Consequence: the IR benchmark is computed per scenario, with the scenario's preferences, as already written in the draft.
  - Solve R and I for every in-sample and out-of-sample scenario, and record which of the two is the binding benchmark for each member.

- **The deterministic baseline is DP with budget balance and soft IR, solved at the expected scenario (no scaling).**
  - No justification needed; it is what the assignment asks for.
  - Use exactly the same bounds, regularization and IR weight as in the stochastic models.
  - State in the report that the expected scenario is not the expected profile, because the scaling is non-linear.

- **The base stochastic model is risk-neutral, with budget balance and IR imposed in expectation.**
  - Required by the assignment; no justification needed.
  - Add the IR slacks and their penalty to the draft's formulation, which still has hard IR.
  - With 1 scenario at the expected values it must reproduce the deterministic baseline. This validation is mandatory.

- **The first variant changes only the objective: a weighted combination of expected cost and CVaR of the cost, at a confidence level of 0.8.**
  - Justify in one sentence: it hedges against high-cost scenarios without adding binaries; with 10 scenarios a level of 0.8 averages the two worst scenarios, while 0.9 keeps only the worst.
  - Risk weight: 0.5 in the base comparison, swept over 0, 0.25, 0.5, 0.75 and 1.
  - With a zero weight it must reproduce the base model; check it.
  - Add the missing reference for the CVaR reformulation (the draft shows "[?]").

- **The second variant changes only the cap: a chance constraint, joint over the 24 hours, implemented with one binary and one SOS1 set per scenario.**
  - Justify in one sentence: one scaling parameter moves all hours of a member together, so a scenario is either compliant or not.
  - A scenario counts as violated if the cap is exceeded in any hour.
  - Violation level: 0.2 in the base comparison.
  - With N equiprobable scenarios, the violation level only changes in steps of 1/N; a level of 0 means no exceedance in any in-sample scenario.
  - Small violation levels may be infeasible; report this as a result, not as an error.
  - The penalty stays in the objective.

- **Only one switch changes per variant; budget balance and IR stay in expectation in all three models.**
  - Consistent with the assignment's rule of one switch at a time; no further justification needed.
  - Per-scenario requirements are an optional additional variant, only if time allows.

- **Out of sample, the realized budget surplus or deficit is absorbed by the manager.**
  - State it in one sentence.
  - Report the distribution of the budget residual, the probability of a deficit and the worst deficit.

- **The IR shortfall is split into a structural part (already present without uncertainty) and a part caused by the uncertainty, with one slack per member. (FA question 1.3)**
  - Use the deterministic baseline's shortfall as the structural part.
  - Report the two parts separately in every IR result, and name the blocking members in each.

- **The in-sample sizes are 1, 3, 5, 10, 15 and 20 scenarios, nested.**
  - Justify going beyond the sizes of the data file (5 and 10): the study of N needs enough points to show convergence.
  - Draw 20 scenarios once with one seed and take the first N; do not redraw for each N.
  - Expect 15 and 20 scenarios to hit the time limit: report the MIP gap and the time, and read their results as approximate.
  - Use 10 scenarios for the comparison of the models and for the sensitivity study.

- **The out-of-sample set has 100 scenarios, drawn with a separate seed and identical for every model and every N.**
  - Use a subset (e.g. 20 scenarios) during development.
  - Report both seeds (in-sample and out-of-sample) in the report and in the README, and save every scenario set with the code.
  - An optional stability check: a second in-sample draw of 10 scenarios with another seed.

- **The out-of-sample response is evaluated twice: optimistic and pessimistic. (FA question 1.5)**
  - The optimistic evaluation is an LP; the pessimistic one needs a small MILP for the meter split.
  - Report both, and compute the value of the stochastic solution for both.
  - This also quantifies, for Q2, how far the community's import can move within the members' best responses, which the feedback asks for.
