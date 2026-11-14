"""Sensitivity analysis for the energy community models."""
from pathlib import Path
import matplotlib.pyplot as plt

from src.model import EnergyCommunityModel
from src.plotting import plot_sensitivity


def _reference_model_cost(com_data):
    """Solve Model R once; its objective is unaffected by contract settings."""
    model = EnergyCommunityModel(com_data, mode="R", verbose=False)
    model.build_model()
    return model.solve()["objective_value"]


def run_discount_sensitivity(com_data, out_dir: Path):
    """Analyze how changing the tariff discount (beta) affects total costs."""
    print("Running Discount Sensitivity...")
    discounts = [0.0, 0.25, 0.5, 0.75, 1.0]
    costs_i, costs_c = [], []
    costs_r = [_reference_model_cost(com_data)] * len(discounts)
    
    for d in discounts:
        # Run Model I
        m_i = EnergyCommunityModel(com_data, mode="I", config={"discount": d}, verbose=False)
        m_i.build_model()
        costs_i.append(m_i.solve()["objective_value"])
        
        # Run Model C
        m_c = EnergyCommunityModel(com_data, mode="C", config={"discount": d}, verbose=False)
        m_c.build_model()
        costs_c.append(m_c.solve()["objective_value"])
        
    fig = plot_sensitivity(
        discounts,
        {
            "Model R (Reference)": costs_r,
            "Model I (Individual)": costs_i,
            "Model C (Community)": costs_c,
        },
        xlabel="Tariff Discount Ratio (\u03b2)",
        title="Sensitivity: Tariff Discount vs Total Cost",
        save_to=out_dir / "Sensitivity_Discount.png",
    )
    plt.close(fig)


def run_penalty_sensitivity(com_data, out_dir: Path):
    """Analyze how changing the penalty price (alpha) affects total costs."""
    print("Running Penalty Sensitivity...")
    penalties = [0.125, 5, 10, 20, 50]  # DKK/kWh
    costs_i, costs_c = [], []
    costs_r = [_reference_model_cost(com_data)] * len(penalties)
    
    for p in penalties:
        m_i = EnergyCommunityModel(com_data, mode="I", config={"penalty": p}, verbose=False)
        m_i.build_model()
        costs_i.append(m_i.solve()["objective_value"])
        
        m_c = EnergyCommunityModel(com_data, mode="C", config={"penalty": p}, verbose=False)
        m_c.build_model()
        costs_c.append(m_c.solve()["objective_value"])
        
    fig = plot_sensitivity(
        penalties,
        {
            "Model R (Reference)": costs_r,
            "Model I (Individual)": costs_i,
            "Model C (Community)": costs_c,
        },
        xlabel="Penalty Price (\u03b1) [DKK/kWh]",
        title="Sensitivity: Penalty Price vs Total Cost",
        save_to=out_dir / "Sensitivity_Penalty.png",
    )
    plt.close(fig)


def run_cap_sensitivity(com_data, out_dir: Path):
    """Analyze how changing the capacity limit tightness affects total costs."""
    print("Running Cap Limit Sensitivity...")
    # 1.0 is the baseline. 0.5 is half the capacity (tighter), 1.5 is loose.
    cap_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    costs_i, costs_c = [], []
    costs_r = [_reference_model_cost(com_data)] * len(cap_factors)
    
    for c in cap_factors:
        m_i = EnergyCommunityModel(com_data, mode="I", config={"cap_factor": c}, verbose=False)
        m_i.build_model()
        costs_i.append(m_i.solve()["objective_value"])
        
        m_c = EnergyCommunityModel(com_data, mode="C", config={"cap_factor": c}, verbose=False)
        m_c.build_model()
        costs_c.append(m_c.solve()["objective_value"])
        
    fig = plot_sensitivity(
        cap_factors,
        {
            "Model R (Reference)": costs_r,
            "Model I (Individual)": costs_i,
            "Model C (Community)": costs_c,
        },
        xlabel="Cap Factor (1.0 = Default Contract Limit)",
        title="Sensitivity: Capacity Limit vs Total Cost",
        save_to=out_dir / "Sensitivity_Cap.png",
    )
    plt.close(fig)
