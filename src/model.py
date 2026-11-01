import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.data_loader import CommunityData

class EnergyCommunityModel:
    """
    General optimization model for Assignment 2.
    Supported modes:
      - "R": Reference (no contract, independent scheduling)
      - "I": Individual (individual contracts, independent scheduling)
      - "C": Central / Community (community contract, central dispatch)
      - "DP": Dynamic Pricing (placeholder for Question 2)
    """

    def __init__(self, data: CommunityData, mode: str = "C", config: dict = None, verbose: bool = False):
        self.data = data
        self.mode = mode.upper()
        self.config = config or {}
        
        # Allow overriding default contract terms via config (useful for sensitivity analysis)
        self.discount = self.config.get("discount", self.data.contract.tariff_discount_ratio)
        self.penalty = self.config.get("penalty", self.data.contract.penalty_DKK_per_kWh)
        # Cap limit can be scaled using a tightness factor
        self.cap_factor = self.config.get("cap_factor", 1.0)
        self.cap_limit = self.data.contract.capacity_limit_kW * self.cap_factor

        self.m = gp.Model(f"EnergyCommunity_{self.mode}")
        self.m.Params.OutputFlag = 1 if verbose else 0
        
        self.T = range(24) # 24-hour horizon
        self.members = self.data.members

        # Dictionaries to store variables and constraints (useful for extracting duals later)
        self.var: Dict[str, Any] = {}
        self.con: Dict[str, Any] = {}

    def build_model(self):
        """Sequentially build variables, constraints, and the objective function."""
        self._build_variables()
        self._build_device_constraints() # Base asset constraints (shared across R, I, C)
        self._build_grid_constraints()   # Grid and contract constraints (differs by mode)
        self._build_objective()          # Objective function (differs by mode)
        self.m.update()

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
                
        elif self.mode == "C":
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
            self.con[f"E_min_{i}"] = self.m.addConstr(gp.quicksum(l[i, t] for t in self.T) == m_data.min_daily_energy_kWh, name=f"E_min_{i}")
            
            # 3. Linearization of absolute deviation for disutility (dev >= |l - l_ref|)
            self.con[f"dev_pos_{i}"] = self.m.addConstrs((dev[i, t] >= l[i, t] - m_data.reference_load[t] for t in self.T), name=f"dev_pos_{i}")
            self.con[f"dev_neg_{i}"] = self.m.addConstrs((dev[i, t] >= m_data.reference_load[t] - l[i, t] for t in self.T), name=f"dev_neg_{i}")
            
            # Explicit non-negativity for deviation
            self.con[f"dev_min_{i}"] = self.m.addConstrs((dev[i, t] >= 0 for t in self.T), name=f"dev_min_{i}")

            # 4. PV generation limits
            self.con[f"pv_min_{i}"] = self.m.addConstrs((g[i, t] >= 0 for t in self.T), name=f"pv_min_{i}")
            self.con[f"pv_max_{i}"] = self.m.addConstrs((g[i, t] <= m_data.pv_available[t] for t in self.T), name=f"pv_max_{i}")

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
                    soc_prev = m_data.battery_initial_soc_kWh if t == 0 else soc[i, t-1]
                    self.con[f"soc_dyn_{i}_{t}"] = self.m.addConstr(
                        soc[i, t] == soc_prev + p_ch[i, t] * m_data.battery_charging_efficiency 
                                     - p_dis[i, t] / m_data.battery_discharging_efficiency,
                        name=f"soc_dyn_{i}_{t}"
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
                    name=f"node_bal_{i}"
                )

                if self.mode == "I":
                    q_below = self.var["q_below"]
                    q_exc = self.var["q_exc"]
                    kappa_i = self.data.contract.share[m_data.consumer_id]
                    
                    # Explicit non-negativity
                    self.con[f"q_below_min_{i}"] = self.m.addConstrs((q_below[i, t] >= 0 for t in self.T), name=f"q_below_min_{i}")
                    self.con[f"q_exc_min_{i}"] = self.m.addConstrs((q_exc[i, t] >= 0 for t in self.T), name=f"q_exc_min_{i}")
                    
                    # Split import into below-cap and exceedance, constrained by individual cap
                    self.con[f"imp_split_{i}"] = self.m.addConstrs((q_imp[i, t] == q_below[i, t] + q_exc[i, t] for t in self.T), name=f"imp_split_{i}")
                    self.con[f"cap_indiv_{i}"] = self.m.addConstrs((q_below[i, t] <= kappa_i * self.cap_limit[t] for t in self.T), name=f"cap_indiv_{i}")

        elif self.mode == "C":
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
                (gp.quicksum(l[i, t] + p_ch[i, t] - g[i, t] - p_dis[i, t] for i in range(len(self.members))) 
                 + P_exp[t] == P_imp[t] for t in self.T),
                name="community_balance"
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
                
        elif self.mode == "C":
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
        self.m.optimize()
        
        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Model did not solve optimally. Status code: {self.m.Status}")
            
        results = {
            "objective_value": self.m.ObjVal,
            "status": "Optimal"
        }
        
        # For Model C, extract the internal clearing prices (Shadow prices / Dual variables)
        if self.mode == "C":
            internal_prices = []
            for t in self.T:
                # Retrieve the Pi attribute (shadow price) of the community balance constraint
                pi = self.con["community_balance"][t].Pi
                internal_prices.append(pi)
            results["internal_prices"] = internal_prices
            
        return results