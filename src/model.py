import math

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.data_loader import CommunityData

DEVICE_KEYS = ("load", "pv", "dev", "p_ch", "p_dis", "soc")


class EnergyCommunityModel:
    """
    General optimization model for Assignment 2.
    Supported modes:
      - "R": Reference (no contract, independent scheduling)
      - "I": Individual (individual contracts, independent scheduling)
      - "C": Central / Community (community contract, central dispatch)
      - "DP": Dynamic Pricing (optimistic KKT/SOS1 reformulation, Question 2)
    """

    def __init__(self, data: CommunityData, mode: str = "C", config: dict = None, verbose: bool = False):
        self.data = data
        self.mode = mode.upper()
        self.config = config or {}
        if self.mode not in {"R", "I", "C", "DP"}:
            raise ValueError(f"Unknown model mode: {self.mode}")
        if self.config.get("daily_energy_rule", "fixed") not in {"fixed", "minimum"}:
            raise ValueError("daily_energy_rule must be 'fixed' or 'minimum'")

        # === [CHANGE 1] Add soft_ir switch ===
        self.soft_ir = bool(self.config.get("soft_ir", False))
        # === [END CHANGE 1] ===

        # Allow overriding default contract terms via config (useful for sensitivity analysis)
        self.discount = self.config.get("discount", self.data.contract.tariff_discount_ratio)
        self.penalty = self.config.get("penalty", self.data.contract.penalty_DKK_per_kWh)
        # Cap limit can be scaled using a tightness factor
        self.cap_factor = self.config.get("cap_factor", 1.0)
        self.cap_limit = self.data.contract.capacity_limit_kW * self.cap_factor

        self.m = gp.Model(f"EnergyCommunity_{self.mode}")
        self.m.Params.OutputFlag = 1 if verbose else 0
        if self.mode == "DP":
            self.m.Params.TimeLimit = self.config.get("time_limit", 300.0)
            self.m.Params.MIPGap = self.config.get("mip_gap", 1e-4)
            if not 0 <= self.discount <= 1 or self.cap_factor < 0:
                raise ValueError("DP requires discount in [0, 1] and nonnegative cap_factor")
            if self.penalty < (1 - self.discount) * self.data.import_tariff:
                raise ValueError("The continuous import split requires penalty >= discounted import tariff")

        self.T = range(24)  # 24-hour horizon
        self.members = self.data.members

        # Dictionaries to store variables and constraints (useful for extracting duals later)
        self.var: Dict[str, Any] = {}
        self.con: Dict[str, Any] = {}

    def build_model(self):
        """Sequentially build variables, constraints, and the objective function."""
        self._build_variables()
        self._build_device_constraints()  # Base asset constraints (shared across R, I, C)
        self.m.update()
        device_rows = list(self.m.getConstrs())
        self._build_grid_constraints()    # Grid and contract constraints (differs by mode)
        self._build_objective()           # Objective function (differs by mode)
        self.m.update()
        if self.mode == "DP":
            self._build_pricing_conditions(device_rows)
            self.m.update()

    def _build_pricing_conditions(self, device_rows):
        """Attach linear stationarity, SOS1 complementarity and settlement switches."""
        m = self.m
        config = self.config
        bounds = config.get("price_bounds", (None, None))
        if len(bounds) != 2:
            raise ValueError("price_bounds must be (lower, upper); use None for an unbounded side")
        lower, upper = bounds
        lower = -GRB.INFINITY if lower is None else float(lower)
        upper = GRB.INFINITY if upper is None else float(upper)
        if math.isnan(lower) or math.isnan(upper) or lower > upper:
            raise ValueError("Invalid price_bounds")
        prices = m.addVars(self.T, lb=lower, ub=upper, name="lambda")
        self.var["lambda"] = prices

        # Settlement constraints can otherwise make wasteful grid loops or voluntary
        # above-cap classification useful for absorbing surplus. Metered flows and
        # exceedance must remain physical, even when budget balance is imposed.
        self.var["cap_slack"] = m.addVars(self.T, lb=0.0, name="unused_cap")
        for t in self.T:
            m.addConstr(
                self.var["cap_slack"][t] == self.cap_limit[t] - self.var["P_below"][t],
                name=f"unused_cap_definition_{t}",
            )
            m.addSOS(GRB.SOS_TYPE1, [self.var["P_imp"][t], self.var["P_exp"][t]])
            m.addSOS(GRB.SOS_TYPE1, [self.var["P_exc"][t], self.var["cap_slack"][t]])

        # All source variables are free; the original device rows contain their bounds.
        index_to_member = {}
        stationarity = {}
        for i, member in enumerate(self.members):
            for key in DEVICE_KEYS:
                for t in self.T:
                    variable = self.var[key][i, t]
                    if variable.LB > -GRB.INFINITY or variable.UB < GRB.INFINITY:
                        raise ValueError("Follower bounds must be explicit device constraints")
                    index_to_member[variable.index] = i
                    stationarity[variable.index] = gp.LinExpr()
        dual_objectives = {i: gp.LinExpr() for i in range(len(self.members))}
        self.member_kkt = []

        # Normalize each inequality to a*x >= b, with dual y >= 0.
        # Equalities use the same orientation and an unrestricted dual.
        for row_constraint in device_rows:
            row = m.getRow(row_constraint)
            indices = [row.getVar(j).index for j in range(row.size())]
            members = {index_to_member[j] for j in indices}
            if len(members) != 1:
                raise ValueError("A follower device row must belong to exactly one member")
            i = members.pop()
            equality = row_constraint.Sense == "="
            sign = -1.0 if row_constraint.Sense == "<" else 1.0
            rhs = sign * row_constraint.RHS

            # Give duals a safe finite upper bound when prices are bounded so Gurobi can
            # convert SOS1 constraints into tight binary formulations with cutting planes.
            dual_ub = float(config.get("dual_ub", max(100.0, 5.0 * upper))) if upper < GRB.INFINITY else GRB.INFINITY
            dual = m.addVar(
                lb=-dual_ub if equality else 0.0,
                ub=dual_ub,
                name=f"dual_{row_constraint.ConstrName}",
            )
            for j in range(row.size()):
                stationarity[indices[j]] += sign * row.getCoeff(j) * dual
            dual_objectives[i] += rhs * dual
            slack = None
            if not equality:
                slack = m.addVar(lb=0.0, name=f"slack_{row_constraint.ConstrName}")
                m.addConstr(
                    slack == sign * row - rhs,
                    name=f"slack_definition_{row_constraint.ConstrName}",
                )
                m.addSOS(GRB.SOS_TYPE1, [dual, slack])
            self.member_kkt.append({
                "member": i, "row": row_constraint,
                "dual": dual, "slack": slack,
            })

        self.member_operating_cost = {}
        self.member_payment = {}
        self.member_total_cost = {}
        self.con["member_stationarity"] = {}
        for i, member in enumerate(self.members):
            self.member_operating_cost[i] = gp.quicksum(
                member.linear_disutility * self.var["dev"][i, t]
                + member.pv_marginal_cost * self.var["pv"][i, t]
                for t in self.T
            )
            for key in DEVICE_KEYS:
                for t in self.T:
                    coefficient = {
                        "load": prices[t], "pv": member.pv_marginal_cost - prices[t],
                        "dev": member.linear_disutility, "p_ch": prices[t],
                        "p_dis": -prices[t], "soc": 0.0,
                    }[key]
                    variable = self.var[key][i, t]
                    self.con["member_stationarity"][i, key, t] = m.addConstr(
                        stationarity[variable.index] == coefficient,
                        name=f"stationarity_{i}_{key}_{t}",
                    )
            # Strong duality at a KKT point: payment = dual value - operating cost.
            self.member_payment[i] = dual_objectives[i] - self.member_operating_cost[i]
            self.member_total_cost[i] = dual_objectives[i]

        # === [CHANGE 2] Save original system-cost objective & add per-member IR slacks ===
        self._system_cost_expr = m.getObjective()
        self.external_grid_cost = self._system_cost_expr - gp.quicksum(
            self.member_operating_cost.values()
        )
        if config.get("budget_balance", False):
            self.con["budget_balance"] = m.addConstr(
                gp.quicksum(self.member_payment.values()) == self.external_grid_cost,
                name="budget_balance",
            )
        if config.get("individual_rationality", False):
            benchmarks = config.get("benchmark_costs")
            ids = {member.consumer_id for member in self.members}
            if benchmarks is None or not ids.issubset(benchmarks):
                raise ValueError("IR requires benchmark_costs mapping every consumer ID to its outside cost")

            if self.soft_ir:
                self.var["ir_slack"] = m.addVars(len(self.members), lb=0.0, name="ir_slack")

            self.con["individual_rationality"] = {}
            for i, member in enumerate(self.members):
                value = float(benchmarks[member.consumer_id])
                if not math.isfinite(value):
                    raise ValueError("Every benchmark cost must be finite")
                rhs = value + self.var["ir_slack"][i] if self.soft_ir else value
                self.con["individual_rationality"][i] = m.addConstr(
                    self.member_total_cost[i] <= rhs,
                    name=f"individual_rationality_{member.consumer_id}",
                )
        # === [END CHANGE 2] ===

    def _build_variables(self):
        """Declare all decision variables with lb=-GRB.INFINITY for KKT dual extraction."""
        # 1. Base variables common to all members
        self.var["load"] = self.m.addVars(len(self.members), 24, lb=-GRB.INFINITY, name="load")
        self.var["pv"] = self.m.addVars(len(self.members), 24, lb=-GRB.INFINITY, name="pv")
        self.var["dev"] = self.m.addVars(len(self.members), 24, lb=-GRB.INFINITY, name="deviation")

        self.var["p_ch"] = self.m.addVars(len(self.members), 24, lb=-GRB.INFINITY, name="p_ch")
        self.var["p_dis"] = self.m.addVars(len(self.members), 24, lb=-GRB.INFINITY, name="p_dis")
        self.var["soc"] = self.m.addVars(len(self.members), 24, lb=-GRB.INFINITY, name="soc")

        # 2. Grid exchange variables based on the active mode
        if self.mode in ["R", "I"]:
            # Modes R and I: Each member has their own metered import/export
            self.var["q_imp"] = self.m.addVars(len(self.members), 24, lb=-GRB.INFINITY, name="q_imp")
            self.var["q_exp"] = self.m.addVars(len(self.members), 24, lb=-GRB.INFINITY, name="q_exp")

            if self.mode == "I":
                # Mode I: Individual import is split into "below cap" and "exceedance"
                self.var["q_below"] = self.m.addVars(len(self.members), 24, lb=-GRB.INFINITY, name="q_below")
                self.var["q_exc"] = self.m.addVars(len(self.members), 24, lb=-GRB.INFINITY, name="q_exc")

        elif self.mode in {"C", "DP"}:
            # Mode C: The community shares a single meter at the connection point
            self.var["P_imp"] = self.m.addVars(24, lb=-GRB.INFINITY, name="P_imp")
            self.var["P_exp"] = self.m.addVars(24, lb=-GRB.INFINITY, name="P_exp")
            self.var["P_below"] = self.m.addVars(24, lb=-GRB.INFINITY, name="P_below")
            self.var["P_exc"] = self.m.addVars(24, lb=-GRB.INFINITY, name="P_exc")

    def _build_device_constraints(self):
        """Build independent device constraints for each member with explicit non-negativity bounds."""
        l = self.var["load"]
        g = self.var["pv"]
        dev = self.var["dev"]
        p_ch = self.var["p_ch"]
        p_dis = self.var["p_dis"]
        soc = self.var["soc"]

        for i, m_data in enumerate(self.members):
            # 1. Flexible load bounds (load_lower_kWh acts as the non-negativity/minimum bound)
            self.con[f"l_min_{i}"] = self.m.addConstrs((l[i, t] >= m_data.load_lower_kWh[t] for t in self.T), name=f"l_min_{i}")
            self.con[f"l_max_{i}"] = self.m.addConstrs((l[i, t] <= m_data.load_upper_kWh[t] for t in self.T), name=f"l_max_{i}")

            # 2. Daily minimum energy requirement
            daily_energy = gp.quicksum(l[i, t] for t in self.T)
            # Preserve the Q1 fixed-energy convention by default; minimum is selectable
            # in every mode so that C and DP always use identical feasible sets.
            energy_constraint = (
                daily_energy >= m_data.min_daily_energy_kWh
                if self.config.get("daily_energy_rule", "fixed") == "minimum"
                else daily_energy == m_data.min_daily_energy_kWh
            )
            self.con[f"E_min_{i}"] = self.m.addConstr(energy_constraint, name=f"E_min_{i}")

            # 3. Linearization of absolute deviation for disutility (dev >= |l - l_ref|)
            # Note: dev >= 0 is implied by dev_pos and dev_neg; omitting it removes 384 redundant SOS1s.
            self.con[f"dev_pos_{i}"] = self.m.addConstrs(
                (dev[i, t] >= l[i, t] - m_data.reference_load[t] for t in self.T), name=f"dev_pos_{i}"
            )
            self.con[f"dev_neg_{i}"] = self.m.addConstrs(
                (dev[i, t] >= m_data.reference_load[t] - l[i, t] for t in self.T), name=f"dev_neg_{i}"
            )

            # 4. PV generation limits (use equality g == 0 when pv_available == 0 to avoid redundant SOS1s)
            for t in self.T:
                if m_data.pv_available[t] <= 0.0:
                    self.con[f"pv_zero_{i}_{t}"] = self.m.addConstr(g[i, t] == 0, name=f"pv_zero_{i}_{t}")
                else:
                    self.con[f"pv_min_{i}_{t}"] = self.m.addConstr(g[i, t] >= 0, name=f"pv_min_{i}_{t}")
                    self.con[f"pv_max_{i}_{t}"] = self.m.addConstr(
                        g[i, t] <= m_data.pv_available[t], name=f"pv_max_{i}_{t}"
                    )

            # 5. Battery constraints
            if m_data.has_battery:
                # Explicit non-negativity for battery operations
                self.con[f"ch_min_{i}"] = self.m.addConstrs((p_ch[i, t] >= 0 for t in self.T), name=f"ch_min_{i}")
                self.con[f"dis_min_{i}"] = self.m.addConstrs((p_dis[i, t] >= 0 for t in self.T), name=f"dis_min_{i}")
                self.con[f"soc_min_{i}"] = self.m.addConstrs((soc[i, t] >= 0 for t in self.T), name=f"soc_min_{i}")

                self.con[f"ch_max_{i}"] = self.m.addConstrs((p_ch[i, t] <= m_data.battery_max_charge_kW for t in self.T), name=f"ch_max_{i}")
                self.con[f"dis_max_{i}"] = self.m.addConstrs((p_dis[i, t] <= m_data.battery_max_discharge_kW for t in self.T), name=f"dis_max_{i}")
                self.con[f"soc_max_{i}"] = self.m.addConstrs((soc[i, t] <= m_data.battery_capacity_kWh for t in self.T), name=f"soc_max_{i}")

                # State of Charge (SoC) dynamics
                for t in self.T:
                    soc_prev = m_data.battery_initial_soc_kWh if t == 0 else soc[i, t - 1]
                    self.con[f"soc_dyn_{i}_{t}"] = self.m.addConstr(
                        soc[i, t] == soc_prev + p_ch[i, t] * m_data.battery_charging_efficiency
                        - p_dis[i, t] / m_data.battery_discharging_efficiency,
                        name=f"soc_dyn_{i}_{t}",
                    )
                # Ensure the final SoC is at least the initial SoC
                self.con[f"soc_end_{i}"] = self.m.addConstr(soc[i, 23] >= m_data.battery_initial_soc_kWh, name=f"soc_end_{i}")
            else:
                # Force battery variables to 0 for members without a battery
                self.con[f"no_ch_{i}"] = self.m.addConstrs((p_ch[i, t] == 0 for t in self.T), name=f"no_ch_{i}")
                self.con[f"no_dis_{i}"] = self.m.addConstrs((p_dis[i, t] == 0 for t in self.T), name=f"no_dis_{i}")
                self.con[f"no_soc_{i}"] = self.m.addConstrs((soc[i, t] == 0 for t in self.T), name=f"no_soc_{i}")

    def _build_grid_constraints(self):
        """Build grid and energy balance constraints based on the contract mode with explicit non-negativity."""
        l = self.var["load"]
        g = self.var["pv"]
        p_ch = self.var["p_ch"]
        p_dis = self.var["p_dis"]

        if self.mode in ["R", "I"]:
            q_imp = self.var["q_imp"]
            q_exp = self.var["q_exp"]

            for i, m_data in enumerate(self.members):
                # Explicit non-negativity for import and export
                self.con[f"q_imp_min_{i}"] = self.m.addConstrs((q_imp[i, t] >= 0 for t in self.T), name=f"q_imp_min_{i}")
                self.con[f"q_exp_min_{i}"] = self.m.addConstrs((q_exp[i, t] >= 0 for t in self.T), name=f"q_exp_min_{i}")

                # Node energy balance: Each member balances their own import/export
                self.con[f"node_bal_{i}"] = self.m.addConstrs(
                    (l[i, t] + p_ch[i, t] + q_exp[i, t] == g[i, t] + p_dis[i, t] + q_imp[i, t] for t in self.T),
                    name=f"node_bal_{i}",
                )

                if self.mode == "I":
                    q_below = self.var["q_below"]
                    q_exc = self.var["q_exc"]
                    individual_cap = (
                        self.data.contract.individual_capacity_limit_kW[m_data.consumer_id]
                        * self.cap_factor
                    )

                    # Explicit non-negativity
                    self.con[f"q_below_min_{i}"] = self.m.addConstrs((q_below[i, t] >= 0 for t in self.T), name=f"q_below_min_{i}")
                    self.con[f"q_exc_min_{i}"] = self.m.addConstrs((q_exc[i, t] >= 0 for t in self.T), name=f"q_exc_min_{i}")

                    # Split import into below-cap and exceedance, constrained by individual cap
                    self.con[f"imp_split_{i}"] = self.m.addConstrs((q_imp[i, t] == q_below[i, t] + q_exc[i, t] for t in self.T), name=f"imp_split_{i}")
                    self.con[f"cap_indiv_{i}"] = self.m.addConstrs((q_below[i, t] <= individual_cap[t] for t in self.T), name=f"cap_indiv_{i}")

        elif self.mode in {"C", "DP"}:
            P_imp = self.var["P_imp"]
            P_exp = self.var["P_exp"]
            P_below = self.var["P_below"]
            P_exc = self.var["P_exc"]

            # Explicit non-negativity for community grid variables
            self.con["P_imp_min"] = self.m.addConstrs((P_imp[t] >= 0 for t in self.T), name="P_imp_min")
            self.con["P_exp_min"] = self.m.addConstrs((P_exp[t] >= 0 for t in self.T), name="P_exp_min")
            self.con["P_below_min"] = self.m.addConstrs((P_below[t] >= 0 for t in self.T), name="P_below_min")
            self.con["P_exc_min"] = self.m.addConstrs((P_exc[t] >= 0 for t in self.T), name="P_exc_min")

            # Community energy balance (Crucial for extracting internal prices / duals later)
            self.con["community_balance"] = self.m.addConstrs(
                (P_imp[t] - P_exp[t]
                 - gp.quicksum(l[i, t] + p_ch[i, t] - g[i, t] - p_dis[i, t]
                               for i in range(len(self.members))) == 0 for t in self.T),
                name="community_balance",
            )

            # Split total community import and apply the community cap
            self.con["P_imp_split"] = self.m.addConstrs((P_imp[t] == P_below[t] + P_exc[t] for t in self.T), name="P_imp_split")
            self.con["cap_community"] = self.m.addConstrs((P_below[t] <= self.cap_limit[t] for t in self.T), name="cap_community")

    def _build_objective(self):
        """Build the objective function to minimize total cost based on the contract mode."""
        obj = gp.LinExpr()
        price = self.data.energy_price
        t_imp = self.data.import_tariff
        t_exp = self.data.export_tariff
        beta = self.discount
        alpha = self.penalty

        # 1. Internal operational costs for all members (disutility + PV marginal cost)
        for i, m_data in enumerate(self.members):
            obj += gp.quicksum(m_data.linear_disutility * self.var["dev"][i, t] for t in self.T)
            obj += gp.quicksum(m_data.pv_marginal_cost * self.var["pv"][i, t] for t in self.T)

        # 2. Grid transaction costs
        if self.mode == "R":
            q_imp = self.var["q_imp"]
            q_exp = self.var["q_exp"]
            for i in range(len(self.members)):
                obj += gp.quicksum((price[t] + t_imp) * q_imp[i, t] - (price[t] - t_exp) * q_exp[i, t] for t in self.T)

        elif self.mode == "I":
            q_imp = self.var["q_imp"]
            q_exp = self.var["q_exp"]
            q_below = self.var["q_below"]
            q_exc = self.var["q_exc"]
            for i in range(len(self.members)):
                obj += gp.quicksum(
                    price[t] * q_imp[i, t]                             # Base energy price
                    - (price[t] - (1 - beta) * t_exp) * q_exp[i, t]    # Export revenue (discounted tariff)
                    + (1 - beta) * t_imp * q_below[i, t]               # Discounted import tariff (below cap)
                    + alpha * q_exc[i, t]                              # Penalty for exceeding the cap
                    for t in self.T
                )

        elif self.mode in {"C", "DP"}:
            P_imp = self.var["P_imp"]
            P_exp = self.var["P_exp"]
            P_below = self.var["P_below"]
            P_exc = self.var["P_exc"]
            obj += gp.quicksum(
                price[t] * P_imp[t]
                - (price[t] - (1 - beta) * t_exp) * P_exp[t]
                + (1 - beta) * t_imp * P_below[t]
                + alpha * P_exc[t]
                for t in self.T
            )

        self.m.setObjective(obj, GRB.MINIMIZE)

    def solve(self):
        """Solve the optimization model and return the results, including dual variables."""
        # === [CHANGE 3] Two-pass lexicographic solve for Soft IR (min sum(s_i), then min cost) ===
        use_soft_ir = (
            self.mode == "DP"
            and self.soft_ir
            and self.config.get("individual_rationality", False)
        )

        if use_soft_ir:
            ir_violation_expr = gp.quicksum(self.var["ir_slack"].values())
            orig_timelimit = self.m.Params.TimeLimit

            # Step 0 (Warm start): solve DP_BB (~4s) to get an immediate feasible KKT + BB point
            self.m.Params.TimeLimit = min(60.0, orig_timelimit)
            self.m.setObjective(self._system_cost_expr, GRB.MINIMIZE)
            self.m.optimize()
            warm_runtime = self.m.Runtime
            if self.m.SolCount > 0:
                for v in self.m.getVars():
                    v.Start = v.X
                benchmarks = self.config["benchmark_costs"]
                for i, member in enumerate(self.members):
                    cost_val = self.member_total_cost[i].getValue()
                    self.var["ir_slack"][i].Start = max(0.0, cost_val - float(benchmarks[member.consumer_id]))

            # Pass 1: minimize total IR violation (starts immediately with the warm-start incumbent)
            self.m.Params.TimeLimit = orig_timelimit
            self.m.Params.MIPFocus = 1  # prioritize improving feasible solutions
            self.m.setObjective(ir_violation_expr, GRB.MINIMIZE)
            self.m.optimize()
            if self.m.SolCount == 0:
                raise RuntimeError(
                    f"Soft IR Pass 1 has no solution. Status code: {self.m.Status} "
                    "(3=infeasible, 4=infeasible or unbounded, 9=time limit)."
                )
            pass1_runtime = self.m.Runtime
            min_ir_violation = self.m.ObjVal

            # Hand Pass 1 solution to Pass 2 as a MIP start
            for v in self.m.getVars():
                v.Start = v.X

            # Pass 2: lock total IR violation at its minimum, then minimize community cost
            tol = float(self.config.get("ir_slack_tol", 1e-6))
            self.con["ir_violation_cap"] = self.m.addConstr(
                ir_violation_expr <= min_ir_violation + tol,
                name="ir_violation_cap",
            )
            self.m.Params.MIPFocus = 0
            self.m.setObjective(self._system_cost_expr, GRB.MINIMIZE)
            self.m.optimize()
            total_runtime = warm_runtime + pass1_runtime + self.m.Runtime

        else:
            self.m.optimize()
            total_runtime = self.m.Runtime
        # === [END CHANGE 3] ===

        if self.m.SolCount == 0:
            raise RuntimeError(
                f"Model has no solution. Status code: {self.m.Status} "
                "(3=infeasible, 4=infeasible or unbounded, 9=time limit)."
            )
        if self.mode != "DP" and self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Model did not solve optimally. Status code: {self.m.Status}")

        results = {
            "objective_value": self.m.ObjVal,
            "status": "Optimal" if self.m.Status == GRB.OPTIMAL else "Feasible incumbent",
            "solver_status": self.m.Status,
            "runtime_seconds": total_runtime,
        }

        # Extract internal clearing prices for Model C before calculating settlements.
        internal_prices = None
        if self.mode == "C":
            internal_prices = []
            for t in self.T:
                # The balance is written as import - export - net demand = 0,
                # so the equality dual is the positive marginal price.
                internal_prices.append(self.con["community_balance"][t].Pi)
            results["internal_prices"] = internal_prices

        if self.mode == "DP":
            internal_prices = [self.var["lambda"][t].X for t in self.T]
            results["internal_prices"] = internal_prices
            results["mip_gap"] = self.m.MIPGap
            results["objective_bound"] = self.m.ObjBound
            # === [CHANGE 4] Report per-member IR slacks and blocking members ===
            if use_soft_ir:
                ir_slacks = {
                    member.consumer_id: self.var["ir_slack"][i].X
                    for i, member in enumerate(self.members)
                }
                results["ir_slacks"] = ir_slacks
                results["total_ir_violation_DKK"] = sum(ir_slacks.values())
                results["blocking_members"] = [
                    cid for cid, s in ir_slacks.items() if s > 1e-5
                ]
            max_dual = max(abs(item["dual"].X) for item in self.member_kkt)
            results["max_kkt_dual"] = max_dual
            # === [END CHANGE 4] ===
            from src.pricing import pricing_diagnostics
            results.update(pricing_diagnostics(self, internal_prices))

        results.update(self._cost_breakdown(internal_prices))

        return results

    def _cost_breakdown(self, internal_prices=None):
        """Return per-consumer costs and the manager settlement balance."""
        consumer_costs = {}
        internal_cost_total = 0.0

        for i, member in enumerate(self.members):
            internal_cost = sum(
                member.linear_disutility * self.var["dev"][i, t].X
                + member.pv_marginal_cost * self.var["pv"][i, t].X
                for t in self.T
            )
            internal_cost_total += internal_cost

            if self.mode == "R":
                grid_cost = sum(
                    (member.energy_price[t] + member.import_tariff) * self.var["q_imp"][i, t].X
                    - (member.energy_price[t] - member.export_tariff) * self.var["q_exp"][i, t].X
                    for t in self.T
                )
                consumer_costs[member.consumer_id] = internal_cost + grid_cost
            elif self.mode == "I":
                grid_cost = sum(
                    member.energy_price[t] * self.var["q_imp"][i, t].X
                    - (member.energy_price[t] - (1 - self.discount) * member.export_tariff) * self.var["q_exp"][i, t].X
                    + (1 - self.discount) * member.import_tariff * self.var["q_below"][i, t].X
                    + self.penalty * self.var["q_exc"][i, t].X
                    for t in self.T
                )
                consumer_costs[member.consumer_id] = internal_cost + grid_cost

        if self.mode in {"C", "DP"}:
            external_grid_cost = sum(
                self.data.energy_price[t] * self.var["P_imp"][t].X
                - (self.data.energy_price[t] - (1 - self.discount) * self.data.export_tariff) * self.var["P_exp"][t].X
                + (1 - self.discount) * self.data.import_tariff * self.var["P_below"][t].X
                + self.penalty * self.var["P_exc"][t].X
                for t in self.T
            )
            settlement = 0.0
            for i, member in enumerate(self.members):
                consumer_settlement = sum(
                    internal_prices[t] * (
                        self.var["load"][i, t].X
                        + self.var["p_ch"][i, t].X
                        - self.var["pv"][i, t].X
                        - self.var["p_dis"][i, t].X
                    )
                    for t in self.T
                )
                consumer_costs[member.consumer_id] = (
                    sum(
                        member.linear_disutility * self.var["dev"][i, t].X
                        + member.pv_marginal_cost * self.var["pv"][i, t].X
                        for t in self.T
                    )
                    + consumer_settlement
                )
                settlement += consumer_settlement
            manager_balance = settlement - external_grid_cost
        else:
            external_grid_cost = sum(consumer_costs.values()) - internal_cost_total
            manager_balance = 0.0

        return {
            "consumer_costs": consumer_costs,
            "external_grid_cost": external_grid_cost,
            "manager_balance": manager_balance,
            "system_cost_check": sum(consumer_costs.values()) - manager_balance,
        }