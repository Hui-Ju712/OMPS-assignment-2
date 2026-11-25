"""Question 3 stochastic differentiated-pricing model and evaluation tools."""
from __future__ import annotations

import copy
import math
import time
from dataclasses import replace
from typing import Any

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from src.data_loader import CommunityData, InputData, load_community
from src.model import DEVICE_KEYS, EnergyCommunityModel
from src import scenario_generation as sg


def _scenario_catalogue_data(data: CommunityData, scenario: dict) -> CommunityData:
    """Apply one generated scenario to the loaded Q2 member structures.
    let scenario_generation.py jason files convert to parameters can be used in the model."""
    members = []
    for member in data.members:
        appliances = scenario["members"][member.consumer_id]
        load = next((value for key, value in appliances.items() if key.startswith("FL_") or "load_id" in value), None)
        if load is None:
            load = next(value for value in appliances.values() if "max_load_KWh_per_hour" in value)
        ref_factors = load.get("reference_load_capacity_factor")
        if ref_factors is None:
            reference = member.reference_load
            lower, upper = member.load_lower_kWh, member.load_upper_kWh
        else:
            reference = member.load_max_kWh * np.asarray(ref_factors, dtype=float)
            deviation = member.max_hourly_deviation_kWh or 0.0
            lower = np.maximum(member.load_min_kWh, reference - deviation)
            upper = np.minimum(member.load_max_kWh, reference + deviation)
        minimum = load.get("min_total_energy_per_day_kWh")
        if minimum is None:
            minimum = member.min_daily_energy_kWh

        pv = next((value for value in appliances.values() if value.get("DER_type") == "PV"), None)
        if pv is None:
            pv_available = member.pv_available
            pv_profile = member.pv_profile
        else:
            pv_profile = np.asarray(pv.get("hourly_profile_ratio", member.pv_profile), dtype=float)
            pv_available = member.pv_max_kW * pv_profile
        members.append(replace(
            member,
            reference_load=None if reference is None else np.asarray(reference, dtype=float),
            min_daily_energy_kWh=None if minimum is None else float(minimum),
            load_lower_kWh=np.asarray(lower, dtype=float),
            load_upper_kWh=np.asarray(upper, dtype=float),
            linear_disutility=(None if load.get("linear_disutility_DKK_per_kWh") is None
                               else float(load["linear_disutility_DKK_per_kWh"])),
            quadratic_disutility=(None if load.get("quadratic_disutility_DKK_per_kWh2") is None
                                  else float(load["quadratic_disutility_DKK_per_kWh2"])),
            consumption_utility=(None if load.get("consumption_utility_DKK_per_kWh") is None
                                 else float(load["consumption_utility_DKK_per_kWh"])),
            pv_profile=np.asarray(pv_profile, dtype=float),
            pv_available=np.asarray(pv_available, dtype=float),
            raw={**member.raw, "load": load, "pv": pv},
        ))
    return replace(data, members=members)


def expected_value_scenario(data: CommunityData, rules: dict | None = None,
                            quantities=("profile",)) -> dict:
    """Build the deterministic scenario from distribution means, not assumed unit scales."""
    rules = rules or sg.load_json("preference_uncertainty_params.json")
    quantities = set(quantities)
    app = data.raw["appliance_params"]
    bus = data.raw["bus_params"]
    app_day, _ = sg.day_data(app, bus, data.raw["weekly_profiles"], data.day)
    consumers = data.raw["params"]
    spec = sg.rules_to_spec(rules, data.member_ids)
    theta = {quantity: {} for quantity in sg.QUANTITIES}
    for quantity in sg.QUANTITIES:
        for cid in data.member_ids:
            rule = spec[quantity][cid]
            theta[quantity][cid] = float(rule.mean if quantity in quantities else 1.0)
    catalogue = {}
    for group, key in (("DERs", "DER_id"), ("loads", "load_id"),
                       ("storages", "storage_id"), ("heat_pumps", "heat_pump_id")):
        catalogue.update({item[key]: item for item in app_day.get(group) or []})
    member_appliances = {}
    for consumer in consumers["consumers"]:
        cid = consumer["consumer_id"]
        member_appliances[cid] = {}
        for aid in consumer["list_appliances"]:
            appliance = catalogue[aid]
            if "load_id" in appliance:
                appliance = sg.apply_scaling(appliance, {q: theta[q][cid] for q in sg.QUANTITIES})
            else:
                appliance = copy.deepcopy(appliance)
            member_appliances[cid][aid] = appliance
    return {"id": "expected", "probability": 1.0, "index": {}, "theta": theta,
            "members": member_appliances, "is_expected_value": True}


def draw_scenario_sets(data: CommunityData, n_in_sample: int, n_oos: int, seed: int,
                       oos_seed: int | None = None, quantities=("profile",),
                       rules: dict | None = None) -> tuple[list[dict], list[dict], dict]:
    """Draw one nested in-sample sequence and an independent common OOS sequence."""
    if n_in_sample < 1 or n_oos < 1:
        raise ValueError("n_in_sample and n_oos must be positive")
    rules = rules or sg.load_json("preference_uncertainty_params.json")
    app, _ = sg.day_data(data.raw["appliance_params"], data.raw["bus_params"],
                         data.raw["weekly_profiles"], data.day)
    community = data.raw["params"]
    is_scenarios = sg.draw_preference_scenarios(
        app, community, rules, n=n_in_sample, seed=int(seed), quantities=quantities,
    )
    oos_seed = int(seed + 1 if oos_seed is None else oos_seed)
    oos_scenarios = sg.draw_preference_scenarios(
        app, community, rules, n=n_oos, seed=oos_seed, quantities=quantities,
    )
    # Keep one equiprobable convention even when a caller later takes a nested prefix.
    for scenarios in (is_scenarios, oos_scenarios):
        probability = 1.0 / len(scenarios)
        for scenario in scenarios:
            scenario["probability"] = probability
    metadata = {"seed": int(seed), "oos_seed": oos_seed, "quantities": list(quantities),
                "in_sample_count": len(is_scenarios), "oos_count": len(oos_scenarios),
                "nested_prefixes": True}
    return is_scenarios, oos_scenarios, metadata


def _prepare_scenarios(data: CommunityData, scenarios: list[dict], config: dict,
                       compute_benchmarks: bool = True) -> list[dict]:
    """Calculate the minimum of R and I for each scenario as benchmark cost"""
    prepared = []
    references = config.get("ir_reference", "both")
    modes = ("R", "I") if references == "both" else (references,)
    if any(mode not in {"R", "I"} for mode in modes):
        raise ValueError("ir_reference must be 'R', 'I', or 'both'")
    common = {key: config[key] for key in
              ("discount", "penalty", "daily_energy_rule", "cap_factor") if key in config}
    for index, scenario in enumerate(scenarios):
        scenario_data = _scenario_catalogue_data(data, scenario)
        benchmark = scenario.get("benchmark_costs")
        if compute_benchmarks and benchmark is None:
            mode_costs = []
            for mode in modes:
                model = EnergyCommunityModel(scenario_data, mode=mode, config=common)
                model.build_model()
                result = model.solve()
                mode_costs.append(result["consumer_costs"])
                model.m.dispose()
            benchmark = {cid: min(costs[cid] for costs in mode_costs) for cid in data.member_ids}
        prepared.append({"id": scenario.get("id", index),
                         "probability": float(scenario.get("probability", 1.0 / len(scenarios))),
                         "theta": scenario.get("theta", {}), "index": scenario.get("index", {}),
                         "members": scenario.get("members", {}), "data": scenario_data,
                         "benchmark_costs": benchmark, "is_expected_value": scenario.get("is_expected_value", False)})
    probabilities = sum(item["probability"] for item in prepared)
    if not math.isclose(probabilities, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"Scenario probabilities must sum to 1, got {probabilities}")
    return prepared


class StochasticPricingModel:
    """Two-stage risk-neutral pricing MPEC; first-stage prices are shared by all scenarios."""

    def __init__(self, data: CommunityData, scenarios: list[dict], config: dict | None = None,
                 verbose: bool = False):
        self.data = data
        self.config = dict(config or {})
        self.objective_type = self.config.get("objective_type", "expected")
        self.cap_constraint_type = self.config.get("cap_constraint_type", "none")
        if self.objective_type != "expected":
            raise NotImplementedError("Only objective_type='expected' is implemented in this foundation")
        if self.cap_constraint_type != "none":
            raise NotImplementedError("Only cap_constraint_type='none' is implemented in this foundation")
        if self.config.get("price_scope", "individual") != "individual":
            raise ValueError("Question 3 requires differentiated individual prices")
        if self.config.get("daily_energy_rule", "fixed") not in {"fixed", "minimum"}:
            raise ValueError("daily_energy_rule must be 'fixed' or 'minimum'")
        self.discount = float(self.config.get("discount", data.contract.tariff_discount_ratio))
        self.penalty = float(self.config.get("penalty", data.contract.penalty_DKK_per_kWh))
        self.cap_factor = float(self.config.get("cap_factor", 1.0))
        if not 0 <= self.discount <= 1 or self.cap_factor < 0:
            raise ValueError("discount must be in [0, 1] and cap_factor nonnegative")
        if self.penalty < (1 - self.discount) * data.import_tariff:
            raise ValueError("Continuous import split requires penalty >= discounted import tariff")
        self.scenarios = _prepare_scenarios(data, scenarios, self.config)
        for scenario in self.scenarios:
            if not scenario["benchmark_costs"]:
                raise ValueError("Every scenario needs Q2 outside-option benchmark costs")
        self.T = range(24)
        self.members = data.members
        self.m = gp.Model("StochasticPricingModel")
        self.m.Params.OutputFlag = 1 if verbose else 0
        self.m.Params.TimeLimit = float(self.config.get("time_limit", 300.0))
        self.m.Params.MIPGap = float(self.config.get("mip_gap", 1e-4))
        self.var: dict[str, Any] = {"scenario": {}}
        self.con: dict[str, Any] = {"scenario": {}}
        self.member_kkt: list[dict] = []
        self.prices = None
        self._built = False

    def build_model(self):
        if self._built:
            return self
        lower, upper = self.config.get("price_bounds", (None, None))
        lower = -GRB.INFINITY if lower is None else float(lower)
        upper = GRB.INFINITY if upper is None else float(upper)
        if math.isnan(lower) or math.isnan(upper) or lower > upper:
            raise ValueError("Invalid price_bounds")
        self.prices = self.m.addVars(len(self.members), 24, lb=lower, ub=upper, name="x_it")
        scenario_costs = []
        weighted_balance = gp.LinExpr()
        weighted_ir = {i: gp.LinExpr() for i in range(len(self.members))}
        for omega, scenario in enumerate(self.scenarios):
            self._build_scenario(omega, scenario)
            probability = scenario["probability"]
            scenario_costs.append(probability * self.var["scenario"][omega]["community_cost"])
            weighted_balance += probability * self.var["scenario"][omega]["budget_balance"]
            for i in range(len(self.members)):
                weighted_ir[i] += probability * self.var["scenario"][omega]["member_total_cost"][i]
        self.m.addConstr(weighted_balance == 0.0, name="expected_budget_balance")
        for i, member in enumerate(self.members):
            benchmark = sum(s["probability"] * float(s["benchmark_costs"][member.consumer_id])
                            for s in self.scenarios)
            self.m.addConstr(weighted_ir[i] <= benchmark, name=f"expected_ir_{member.consumer_id}")
        self.m.setObjective(gp.quicksum(scenario_costs), GRB.MINIMIZE)
        self.m.update()
        self._built = True
        return self

    def _build_scenario(self, omega: int, scenario: dict):
        m = self.m
        scenario_data = scenario["data"]
        prefix = f"w{omega}"
        v = {}
        c = {}
        self.var["scenario"][omega] = v
        self.con["scenario"][omega] = c
        for key, name in (("load", "load"), ("pv", "pv"), ("dev", "deviation"),
                          ("p_ch", "charge"), ("p_dis", "discharge"), ("soc", "soc")):
            v[key] = m.addVars(len(self.members), 24, lb=-GRB.INFINITY, name=f"{prefix}_{name}")
        v["P_imp"] = m.addVars(24, lb=0.0, name=f"{prefix}_P_imp")
        v["P_exp"] = m.addVars(24, lb=0.0, name=f"{prefix}_P_exp")
        v["P_below"] = m.addVars(24, lb=0.0, name=f"{prefix}_P_below")
        v["P_exc"] = m.addVars(24, lb=0.0, name=f"{prefix}_P_exc")
        v["cap_slack"] = m.addVars(24, lb=0.0, name=f"{prefix}_unused_cap")
        for i, member in enumerate(scenario_data.members):
            c[f"l_min_{i}"] = m.addConstrs((v["load"][i, t] >= member.load_lower_kWh[t] for t in self.T), name=f"{prefix}_l_min_{i}")
            c[f"l_max_{i}"] = m.addConstrs((v["load"][i, t] <= member.load_upper_kWh[t] for t in self.T), name=f"{prefix}_l_max_{i}")
            energy = gp.quicksum(v["load"][i, t] for t in self.T)
            minimum_rule = self.config.get("daily_energy_rule", "fixed") == "minimum"
            if member.min_daily_energy_kWh is not None:
                c[f"E_min_{i}"] = m.addConstr(
                    energy >= member.min_daily_energy_kWh if minimum_rule else energy == member.min_daily_energy_kWh,
                    name=f"{prefix}_E_min_{i}")
            if member.reference_load is not None:
                c[f"dev_pos_{i}"] = m.addConstrs((v["dev"][i, t] >= v["load"][i, t] - member.reference_load[t] for t in self.T), name=f"{prefix}_dev_pos_{i}")
                c[f"dev_neg_{i}"] = m.addConstrs((v["dev"][i, t] >= member.reference_load[t] - v["load"][i, t] for t in self.T), name=f"{prefix}_dev_neg_{i}")
            else:
                c[f"dev_zero_{i}"] = m.addConstrs((v["dev"][i, t] == 0 for t in self.T), name=f"{prefix}_dev_zero_{i}")
            pv_available = member.pv_available if member.pv_available is not None else np.zeros(24)
            for t in self.T:
                if pv_available[t] <= 0.0:
                    c[f"pv_zero_{i}_{t}"] = m.addConstr(
                        v["pv"][i, t] == 0, name=f"{prefix}_pv_zero_{i}_{t}")
                else:
                    c[f"pv_min_{i}_{t}"] = m.addConstr(
                        v["pv"][i, t] >= 0, name=f"{prefix}_pv_min_{i}_{t}")
                    c[f"pv_max_{i}_{t}"] = m.addConstr(
                        v["pv"][i, t] <= pv_available[t], name=f"{prefix}_pv_max_{i}_{t}")
            if member.has_battery:
                for key, upper_bound in (("p_ch", member.battery_max_charge_kW),
                                         ("p_dis", member.battery_max_discharge_kW),
                                         ("soc", member.battery_capacity_kWh)):
                    c[f"{key}_min_{i}"] = m.addConstrs((v[key][i, t] >= 0 for t in self.T), name=f"{prefix}_{key}_min_{i}")
                    c[f"{key}_max_{i}"] = m.addConstrs((v[key][i, t] <= upper_bound for t in self.T), name=f"{prefix}_{key}_max_{i}")
                for t in self.T:
                    soc_previous = member.battery_initial_soc_kWh if t == 0 else v["soc"][i, t - 1]
                    c[f"soc_dyn_{i}_{t}"] = m.addConstr(
                        v["soc"][i, t] == soc_previous + v["p_ch"][i, t] * member.battery_charging_efficiency
                        - v["p_dis"][i, t] / member.battery_discharging_efficiency,
                        name=f"{prefix}_soc_dyn_{i}_{t}")
                c[f"soc_end_{i}"] = m.addConstr(v["soc"][i, 23] >= member.battery_initial_soc_kWh, name=f"{prefix}_soc_end_{i}")
            else:
                for key in ("p_ch", "p_dis", "soc"):
                    c[f"{key}_zero_{i}"] = m.addConstrs((v[key][i, t] == 0 for t in self.T), name=f"{prefix}_{key}_zero_{i}")

        device_rows = [constraint for group in c.values() if isinstance(group, gp.tupledict)
                       for constraint in group.values()]
        device_rows += [group for group in c.values() if isinstance(group, gp.Constr)]
        cap = scenario_data.contract.capacity_limit_kW * self.cap_factor
        for t in self.T:
            c["community_balance", t] = m.addConstr(
                gp.quicksum(v["load"][i, t] + v["p_ch"][i, t] - v["pv"][i, t] - v["p_dis"][i, t]
                            for i in range(len(self.members))) == v["P_imp"][t] - v["P_exp"][t],
                name=f"{prefix}_community_balance_{t}")
            c["import_split", t] = m.addConstr(v["P_imp"][t] == v["P_below"][t] + v["P_exc"][t], name=f"{prefix}_import_split_{t}")
            c["cap", t] = m.addConstr(v["P_below"][t] <= cap[t], name=f"{prefix}_cap_{t}")
            c["unused_cap", t] = m.addConstr(v["cap_slack"][t] == cap[t] - v["P_below"][t], name=f"{prefix}_unused_cap_{t}")
            m.addSOS(GRB.SOS_TYPE1, [v["P_imp"][t], v["P_exp"][t]])
            m.addSOS(GRB.SOS_TYPE1, [v["P_exc"][t], v["cap_slack"][t]])

        m.update()
        index_to_member = {}
        stationarity = {}
        for i in range(len(self.members)):
            for key in DEVICE_KEYS:
                for t in self.T:
                    variable = v[key][i, t]
                    index_to_member[variable.index] = i
                    stationarity[variable.index] = gp.LinExpr()
        dual_objectives = {i: gp.LinExpr() for i in range(len(self.members))}
        for constraint in device_rows:
            row = m.getRow(constraint)
            indices = [row.getVar(j).index for j in range(row.size())]
            member_ids = {index_to_member[index] for index in indices if index in index_to_member}
            if len(member_ids) != 1 or len(member_ids) != len({index_to_member[index] for index in indices if index in index_to_member}):
                raise ValueError(f"Follower row {constraint.ConstrName} must belong to one member")
            i = member_ids.pop()
            equality = constraint.Sense == "="
            sign = -1.0 if constraint.Sense == "<" else 1.0
            rhs = sign * constraint.RHS
            dual = m.addVar(lb=-GRB.INFINITY if equality else 0.0, name=f"dual_{constraint.ConstrName}")
            for j in range(row.size()):
                index = row.getVar(j).index
                if index in stationarity:
                    stationarity[index] += sign * row.getCoeff(j) * dual
            dual_objectives[i] += rhs * dual
            slack = None
            if not equality:
                slack = m.addVar(lb=0.0, name=f"slack_{constraint.ConstrName}")
                m.addConstr(slack == sign * row - rhs, name=f"slack_definition_{constraint.ConstrName}")
                m.addSOS(GRB.SOS_TYPE1, [dual, slack])
            self.member_kkt.append({"scenario": omega, "member": i, "row": constraint,
                                    "dual": dual, "slack": slack})

        operating = {}
        payment = {}
        total_cost = {}
        stationarity_constraints = {}
        for i, member in enumerate(scenario_data.members):
            operating[i] = gp.quicksum(
                (member.linear_disutility or 0.0) * v["dev"][i, t]
                + member.pv_marginal_cost * v["pv"][i, t] for t in self.T)
            for key in DEVICE_KEYS:
                for t in self.T:
                    price_coefficient = {
                        "load": self.prices[i, t],
                        "pv": member.pv_marginal_cost - self.prices[i, t],
                        "dev": member.linear_disutility or 0.0,
                        "p_ch": self.prices[i, t], "p_dis": -self.prices[i, t], "soc": 0.0,
                    }[key]
                    stationarity_constraints[i, key, t] = m.addConstr(
                        stationarity[v[key][i, t].index] == price_coefficient,
                        name=f"{prefix}_stationarity_{i}_{key}_{t}")
            payment[i] = dual_objectives[i] - operating[i]
            total_cost[i] = dual_objectives[i]
        price = scenario_data.energy_price
        import_tariff, export_tariff = scenario_data.import_tariff, scenario_data.export_tariff
        external = gp.quicksum(
            price[t] * v["P_imp"][t]
            - (price[t] - (1 - self.discount) * export_tariff) * v["P_exp"][t]
            + (1 - self.discount) * import_tariff * v["P_below"][t]
            + self.penalty * v["P_exc"][t] for t in self.T)
        total_operating = gp.quicksum(operating.values())
        community_cost = external + total_operating
        budget_balance = gp.quicksum(payment.values()) - external
        v.update({"operating_cost": operating, "payment": payment, "member_total_cost": total_cost,
                  "external_grid_cost": external, "community_cost": community_cost,
                  "budget_balance": budget_balance, "stationarity": stationarity_constraints,
                  "dual_objectives": dual_objectives})

    def solve(self) -> dict:
        if not self._built:
            self.build_model()
        started = time.perf_counter()
        self.m.optimize()
        elapsed = time.perf_counter() - started
        if self.m.SolCount == 0:
            raise RuntimeError(f"Stochastic pricing model has no incumbent (Gurobi status {self.m.Status})")
        diagnostics = self.diagnostics()
        prices = {member.consumer_id: [self.prices[i, t].X for t in self.T]
                  for i, member in enumerate(self.members)}
        in_sample = []
        for omega, scenario in enumerate(self.scenarios):
            v = self.var["scenario"][omega]
            in_sample.append({"id": scenario["id"], "probability": scenario["probability"],
                              "theta": scenario["theta"],
                              "benchmark_costs_DKK": scenario["benchmark_costs"],
                              "community_cost_DKK": v["community_cost"].getValue(),
                              "external_grid_cost_DKK": v["external_grid_cost"].getValue(),
                              "budget_balance_DKK": v["budget_balance"].getValue(),
                              "member_costs_DKK": {self.members[i].consumer_id: v["member_total_cost"][i].getValue()
                                                    for i in range(len(self.members))},
                              "imports_kW": [v["P_imp"][t].X for t in self.T],
                              "exports_kW": [v["P_exp"][t].X for t in self.T],
                              "exceedance_kW": [v["P_exc"][t].X for t in self.T]})
        return {"status": "Optimal" if self.m.Status == GRB.OPTIMAL else "Feasible incumbent",
                "objective_value_DKK": self.m.ObjVal,
                "objective_bound_DKK": self.m.ObjBound,
                "mip_gap": self.m.MIPGap if self.m.IsMIP else 0.0,
                "runtime_seconds": self.m.Runtime,
                "wall_runtime_seconds": elapsed,
                "model_size": self.model_size(), "prices_x_it": prices,
                "in_sample": in_sample, "diagnostics": diagnostics}

    def model_size(self) -> dict:
        return {"variables_continuous": sum(v.VType == GRB.CONTINUOUS for v in self.m.getVars()),
                "variables_binary": sum(v.VType == GRB.BINARY for v in self.m.getVars()),
                "variables_integer": sum(v.VType == GRB.INTEGER for v in self.m.getVars()),
                "linear_constraints": self.m.NumConstrs,
                "sos1_constraints": self.m.NumSOS}

    def diagnostics(self) -> dict:
        primal = stationarity_residual = complementarity = payment_residual = 0.0
        for entry in self.member_kkt:
            row = entry["row"]
            residual = self.m.getRow(row).getValue() - row.RHS
            violation = abs(residual) if row.Sense == "=" else (
                max(0.0, residual) if row.Sense == "<" else max(0.0, -residual))
            primal = max(primal, violation)
            if entry["slack"] is not None:
                complementarity = max(complementarity, abs(entry["dual"].X * entry["slack"].X))
        for omega, v in self.var["scenario"].items():
            stationarity_residual = max(stationarity_residual, max(
                (abs(row.Slack) for row in v["stationarity"].values()), default=0.0))
            scenario_data = self.scenarios[omega]["data"]
            for i in range(len(scenario_data.members)):
                direct = sum(self.prices[i, t].X * (
                    v["load"][i, t].X + v["p_ch"][i, t].X - v["pv"][i, t].X - v["p_dis"][i, t].X)
                    for t in self.T)
                payment_residual = max(payment_residual, abs(direct - v["payment"][i].getValue()))
        import_export = max((min(v["P_imp"][t].X, v["P_exp"][t].X)
                             for v in self.var["scenario"].values() for t in self.T), default=0.0)
        exceedance = max((abs(v["P_exc"][t].X - max(0.0, v["P_imp"][t].X -
                           self.scenarios[omega]["data"].contract.capacity_limit_kW[t] * self.cap_factor))
                          for omega, v in self.var["scenario"].items() for t in self.T), default=0.0)
        return {"kkt_primal_residual": primal,
                "kkt_stationarity_residual": stationarity_residual,
                "kkt_complementarity_residual": complementarity,
                "payment_identity_residual": payment_residual,
                "max_simultaneous_import_export_kW": import_export,
                "max_fictitious_exceedance_kW": exceedance}

    def dispose(self):
        self.m.dispose()


def _solve_member_lp(member: InputData, prices: np.ndarray, daily_energy_rule: str) -> dict:
    """Solve one member's fixed-price best response as a continuous LP."""
    model = gp.Model("Q3_member_response")
    model.Params.OutputFlag = 0
    hours = range(24)
    load = model.addVars(24, lb=-GRB.INFINITY, name="load")
    pv = model.addVars(24, lb=-GRB.INFINITY, name="pv")
    dev = model.addVars(24, lb=-GRB.INFINITY, name="dev")
    charge = model.addVars(24, lb=-GRB.INFINITY, name="charge")
    discharge = model.addVars(24, lb=-GRB.INFINITY, name="discharge")
    soc = model.addVars(24, lb=-GRB.INFINITY, name="soc")
    model.addConstrs((load[t] >= member.load_lower_kWh[t] for t in hours))
    model.addConstrs((load[t] <= member.load_upper_kWh[t] for t in hours))
    if member.min_daily_energy_kWh is not None:
        energy = gp.quicksum(load[t] for t in hours)
        model.addConstr(energy >= member.min_daily_energy_kWh if daily_energy_rule == "minimum"
                        else energy == member.min_daily_energy_kWh)
    if member.reference_load is not None:
        model.addConstrs((dev[t] >= load[t] - member.reference_load[t] for t in hours))
        model.addConstrs((dev[t] >= member.reference_load[t] - load[t] for t in hours))
    else:
        model.addConstrs((dev[t] == 0 for t in hours))
    if np.max(member.pv_available) > 0:
        model.addConstrs((pv[t] >= 0 for t in hours))
        model.addConstrs((pv[t] <= member.pv_available[t] for t in hours))
    else:
        model.addConstrs((pv[t] == 0 for t in hours))
    if member.has_battery:
        model.addConstrs((charge[t] >= 0 for t in hours))
        model.addConstrs((charge[t] <= member.battery_max_charge_kW for t in hours))
        model.addConstrs((discharge[t] >= 0 for t in hours))
        model.addConstrs((discharge[t] <= member.battery_max_discharge_kW for t in hours))
        model.addConstrs((soc[t] >= 0 for t in hours))
        model.addConstrs((soc[t] <= member.battery_capacity_kWh for t in hours))
        for t in hours:
            previous = member.battery_initial_soc_kWh if t == 0 else soc[t - 1]
            model.addConstr(soc[t] == previous + charge[t] * member.battery_charging_efficiency
                            - discharge[t] / member.battery_discharging_efficiency)
        model.addConstr(soc[23] >= member.battery_initial_soc_kWh)
    else:
        for variable in (charge, discharge, soc):
            model.addConstrs((variable[t] == 0 for t in hours))
    operating = gp.quicksum((member.linear_disutility or 0.0) * dev[t]
                            + member.pv_marginal_cost * pv[t] for t in hours)
    payment_expr = gp.quicksum(float(prices[t]) * (load[t] + charge[t] - pv[t] - discharge[t]) for t in hours)
    model.setObjective(operating + payment_expr, GRB.MINIMIZE)
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(f"Member response LP infeasible for {member.consumer_id}")
    net = np.asarray([load[t].X + charge[t].X - pv[t].X - discharge[t].X for t in hours])
    disutility = sum((member.linear_disutility or 0.0) * dev[t].X for t in hours)
    pv_cost = sum(member.pv_marginal_cost * pv[t].X for t in hours)
    payment = float(np.dot(prices, net))
    result = {"net_consumption_kWh": net.tolist(), "payment_DKK": payment,
              "disutility_DKK": disutility, "pv_cost_DKK": pv_cost,
              "total_cost_DKK": payment + disutility + pv_cost,
              "load_kWh": [load[t].X for t in hours], "pv_kWh": [pv[t].X for t in hours],
              "charge_kWh": [charge[t].X for t in hours],
              "discharge_kWh": [discharge[t].X for t in hours],
              "deviation_kWh": [dev[t].X for t in hours]}
    model.dispose()
    return result


def _distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {key: None for key in ("mean", "std", "min", "max", "var_90", "cvar_90", "var_95", "cvar_95")}
    out = {"mean": float(np.mean(array)), "std": float(np.std(array)),
           "min": float(np.min(array)), "max": float(np.max(array))}
    for level in (0.90, 0.95):
        threshold = float(np.quantile(array, level))
        tail_count = max(1, int(math.ceil((1.0 - level) * len(array))))
        out[f"var_{int(level * 100)}"] = threshold
        out[f"cvar_{int(level * 100)}"] = float(np.mean(np.sort(array)[-tail_count:]))
    return out


def _solve_optimistic_community_lp(data: CommunityData, prices: np.ndarray,
                                   daily_energy_rule: str, cap_factor: float,
                                   discount: float, penalty: float,
                                   individual_responses: dict[str, dict]) -> dict:
    """Choose the community-best dispatch among members' fixed-price best responses."""
    config = {"daily_energy_rule": daily_energy_rule, "cap_factor": cap_factor,
              "discount": discount, "penalty": penalty}
    model = EnergyCommunityModel(data, mode="C", config=config)
    model.build_model()
    for i, member in enumerate(data.members):
        response_cost = float(individual_responses[member.consumer_id]["total_cost_DKK"])
        operating = gp.quicksum(
            (member.linear_disutility or 0.0) * model.var["dev"][i, t]
            + member.pv_marginal_cost * model.var["pv"][i, t]
            for t in model.T
        )
        payment = gp.quicksum(
            float(prices[i, t]) * (
                model.var["load"][i, t] + model.var["p_ch"][i, t]
                - model.var["pv"][i, t] - model.var["p_dis"][i, t]
            )
            for t in model.T
        )
        model.m.addConstr(
            operating + payment <= response_cost + 1e-7,
            name=f"best_response_{member.consumer_id}",
        )
    model.m.optimize()
    if model.m.SolCount == 0:
        status = model.m.Status
        model.m.dispose()
        raise RuntimeError(f"Optimistic community LP has no solution (Gurobi status {status})")

    member_results = {}
    net = np.zeros(24, dtype=float)
    total_operating = 0.0
    member_total = {}
    for i, member in enumerate(data.members):
        load = np.asarray([model.var["load"][i, t].X for t in model.T])
        pv = np.asarray([model.var["pv"][i, t].X for t in model.T])
        deviation = np.asarray([model.var["dev"][i, t].X for t in model.T])
        charge = np.asarray([model.var["p_ch"][i, t].X for t in model.T])
        discharge = np.asarray([model.var["p_dis"][i, t].X for t in model.T])
        member_net = load + charge - pv - discharge
        disutility = float((member.linear_disutility or 0.0) * deviation.sum())
        pv_cost = float(member.pv_marginal_cost * pv.sum())
        payment = float(np.dot(prices[i], member_net))
        member_cost = payment + disutility + pv_cost
        total_operating += disutility + pv_cost
        net += member_net
        member_total[member.consumer_id] = member_cost
        member_results[member.consumer_id] = {
            "net_consumption_kWh": member_net.tolist(), "payment_DKK": payment,
            "disutility_DKK": disutility, "pv_cost_DKK": pv_cost,
            "total_cost_DKK": member_cost, "load_kWh": load.tolist(),
            "pv_kWh": pv.tolist(), "charge_kWh": charge.tolist(),
            "discharge_kWh": discharge.tolist(), "deviation_kWh": deviation.tolist(),
        }

    imports = np.asarray([model.var["P_imp"][t].X for t in model.T])
    exports = np.asarray([model.var["P_exp"][t].X for t in model.T])
    below = np.asarray([model.var["P_below"][t].X for t in model.T])
    exceedance = np.asarray([model.var["P_exc"][t].X for t in model.T])
    cap = data.contract.capacity_limit_kW * cap_factor
    external = float(np.sum(
        data.energy_price * imports
        - (data.energy_price - (1 - discount) * data.export_tariff) * exports
        + (1 - discount) * data.import_tariff * below + penalty * exceedance
    ))
    community_cost = external + total_operating
    max_simultaneous = float(np.max(np.minimum(imports, exports)))
    max_fictitious = float(np.max(np.abs(exceedance - np.maximum(0.0, imports - cap))))
    feasibility_tolerance = 1e-6
    if max_simultaneous > feasibility_tolerance or max_fictitious > feasibility_tolerance:
        model.m.dispose()
        raise RuntimeError(
            "Optimistic community LP returned a nonphysical meter split: "
            f"simultaneous import/export={max_simultaneous:g} kW, "
            f"fictitious exceedance={max_fictitious:g} kW"
        )
    model.m.dispose()
    return {"members": member_results, "net": net, "imports": imports,
            "exports": exports, "exceedance": exceedance,
            "external_grid_cost_DKK": external,
            "total_operating_cost_DKK": total_operating,
            "community_cost_DKK": community_cost,
            "member_costs_DKK": member_total,
            "max_simultaneous_import_export_kW": max_simultaneous,
            "max_fictitious_exceedance_kW": max_fictitious}


def evaluate_prices_on_scenarios(prices_x_it: dict | np.ndarray, scenarios: list[dict],
                                 data: CommunityData, config: dict | None = None,
                                 compute_benchmarks: bool = True,
                                 tie_breaking: str = "optimistic") -> dict:
    """Evaluate fixed prices using optimistic or independent member best responses."""
    config = dict(config or {})
    if tie_breaking not in {"optimistic", "independent"}:
        raise ValueError("tie_breaking must be 'optimistic' or 'independent'")
    prepared = _prepare_scenarios(data, scenarios, config, compute_benchmarks=compute_benchmarks)
    if isinstance(prices_x_it, dict):
        prices = np.asarray([prices_x_it[member.consumer_id] for member in data.members], dtype=float)
    else:
        prices = np.asarray(prices_x_it, dtype=float)
    if prices.shape != (len(data.members), 24):
        raise ValueError(f"prices_x_it must have shape ({len(data.members)}, 24); got {prices.shape}")
    if not np.isfinite(prices).all():
        raise ValueError("prices_x_it must be finite")
    cap_factor = float(config.get("cap_factor", 1.0))
    discount = float(config.get("discount", data.contract.tariff_discount_ratio))
    penalty = float(config.get("penalty", data.contract.penalty_DKK_per_kWh))
    daily_rule = config.get("daily_energy_rule", "fixed")
    rows = []
    for scenario in prepared:
        scenario_data = scenario["data"]
        individual_responses = {
            member.consumer_id: _solve_member_lp(member, prices[i], daily_rule)
            for i, member in enumerate(scenario_data.members)
        }
        if tie_breaking == "optimistic":
            dispatch = _solve_optimistic_community_lp(
                scenario_data, prices, daily_rule, cap_factor, discount, penalty,
                individual_responses,
            )
            members = dispatch["members"]
            net = dispatch["net"]
            imports = dispatch["imports"]
            exports = dispatch["exports"]
            exceedance = dispatch["exceedance"]
            external = dispatch["external_grid_cost_DKK"]
            total_operating = dispatch["total_operating_cost_DKK"]
            community_cost = dispatch["community_cost_DKK"]
            member_total = dispatch["member_costs_DKK"]
            physical_checks = {
                "max_simultaneous_import_export_kW": dispatch["max_simultaneous_import_export_kW"],
                "max_fictitious_exceedance_kW": dispatch["max_fictitious_exceedance_kW"],
            }
        else:
            members = individual_responses
            net = np.sum([members[member.consumer_id]["net_consumption_kWh"]
                          for member in data.members], axis=0)
            imports = np.maximum(net, 0.0)
            exports = np.maximum(-net, 0.0)
            cap = scenario_data.contract.capacity_limit_kW * cap_factor
            below = np.minimum(imports, cap)
            exceedance = np.maximum(imports - cap, 0.0)
            external = float(np.sum(
                scenario_data.energy_price * imports
                - (scenario_data.energy_price - (1 - discount) * scenario_data.export_tariff) * exports
                + (1 - discount) * scenario_data.import_tariff * below + penalty * exceedance))
            total_operating = sum(m["disutility_DKK"] + m["pv_cost_DKK"] for m in members.values())
            community_cost = external + total_operating
            member_total = {cid: m["total_cost_DKK"] for cid, m in members.items()}
            physical_checks = {
                "max_simultaneous_import_export_kW": 0.0,
                "max_fictitious_exceedance_kW": 0.0,
            }
        ir_violations = {cid: max(0.0, member_total[cid] - float(scenario["benchmark_costs"][cid]))
                         for cid in data.member_ids}
        rows.append({"scenario_id": scenario["id"], "probability": scenario["probability"],
                 "theta": scenario["theta"],
                     "community_cost_DKK": community_cost,
                     "external_grid_cost_DKK": external,
                     "total_operating_cost_DKK": total_operating,
                     "budget_balance_DKK": sum(member_total.values()) - community_cost,
                     "member_costs_DKK": member_total,
                     "ir_violations_DKK": ir_violations,
                     "net_community_consumption_kWh": net.tolist(),
                     "imports_kW": imports.tolist(), "exports_kW": exports.tolist(),
                     "exceedance_kW": exceedance.tolist(),
                     "exceedance_kWh": float(np.sum(exceedance)),
                     "physical_checks": physical_checks,
                     "members": members})
    costs = [row["community_cost_DKK"] for row in rows]
    exceedance_energy = [row["exceedance_kWh"] for row in rows]
    balances = [row["budget_balance_DKK"] for row in rows]
    ir_magnitudes = [value for row in rows for value in row["ir_violations_DKK"].values()]
    ir_scenario_violations = [any(value > 1e-6 for value in row["ir_violations_DKK"].values()) for row in rows]
    probabilities = np.asarray([row["probability"] for row in rows], dtype=float)
    probabilities /= probabilities.sum()
    expected_member_costs = {
        cid: float(sum(probabilities[index] * row["member_costs_DKK"][cid]
                       for index, row in enumerate(rows)))
        for cid in data.member_ids
    }
    expected_member_benchmarks = {
        cid: float(sum(probabilities[index] * float(prepared[index]["benchmark_costs"][cid])
                       for index in range(len(rows))))
                   for cid in data.member_ids
    }
    expected_ir_violations = {
        cid: max(0.0, expected_member_costs[cid] - expected_member_benchmarks[cid])
        for cid in data.member_ids
    }
    bb_deficit_frequency = float(np.mean([row["budget_balance_DKK"] < -1e-6 for row in rows]))
    bb_mean = float(np.mean(balances)) if balances else 0.0
    bb_min = float(np.min(balances)) if balances else 0.0
    max_expected_ir_violation = max(expected_ir_violations.values(), default=0.0)
    summary = {"community_cost_DKK": _distribution(costs),
               "cap_exceedance": {"scenario_frequency": float(np.mean([v > 1e-6 for v in exceedance_energy])),
                                  "mean_magnitude_kWh": float(np.mean(exceedance_energy)),
                                  "max_magnitude_kWh": float(np.max(exceedance_energy))},
               "budget_balance_DKK": _distribution(balances),
               "budget_balance_violation_frequency": float(np.mean([abs(v) > 1e-6 for v in balances])),
               "bb_deficit_frequency": bb_deficit_frequency,
               "bb_mean_DKK": bb_mean,
               "bb_min_DKK": bb_min,
               "ir": {"scenario_violation_frequency": float(np.mean(ir_scenario_violations)),
                     "member_scenario_violation_frequency": {
                         cid: float(np.mean([row["ir_violations_DKK"][cid] > 1e-6 for row in rows]))
                         for cid in data.member_ids},
                     "expected_member_cost_DKK": expected_member_costs,
                     "expected_benchmark_cost_DKK": expected_member_benchmarks,
                     "expected_ir_violation_DKK": expected_ir_violations,
                     "max_expected_ir_violation_DKK": max_expected_ir_violation,
                     "mean_violation_DKK": float(np.mean(ir_magnitudes)) if ir_magnitudes else 0.0,
                     "max_violation_DKK": float(np.max(ir_magnitudes)) if ir_magnitudes else 0.0}}
    return {"scenario_count": len(rows), "scenarios": rows, "summary": summary}


def make_model_config(data: CommunityData, price_min: float | None = 0.01,
                      price_max: float | None = 20.0, **overrides) -> dict:
    """Create a Q3 configuration with Q2 contract defaults and price-bound semantics."""
    lower = 0.01 if price_min is None else float(price_min)
    upper = 20.0 if price_max is None else float(price_max)
    config = {"objective_type": "expected", "cap_constraint_type": "none",
              "price_scope": "individual", "price_bounds": (lower, upper),
              "discount": data.contract.tariff_discount_ratio,
              "penalty": data.contract.penalty_DKK_per_kWh,
              "cap_factor": 1.0, "daily_energy_rule": "fixed",
              "ir_reference": "both", "time_limit": 300.0, "mip_gap": 1e-4}
    config.update(overrides)
    return config


def load_q3_data(community: str = "base", day: str = "d1") -> CommunityData:
    return load_community(community, day)
