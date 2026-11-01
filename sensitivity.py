"""Sensitivity analysis for the energy community models."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import load_community
from src.model import EnergyCommunityModel

RESULTS_DIR = Path(__file__).resolve().parent / "results"

def run_discount_sensitivity(com_data, out_dir: Path):
    """Analyze how changing the tariff discount (beta) affects total costs."""
    print("Running Discount Sensitivity...")
    discounts = [0.0, 0.25, 0.5, 0.75, 1.0]
    costs_i, costs_c = [], []
    
    for d in discounts:
        # Run Model I
        m_i = EnergyCommunityModel(com_data, mode="I", config={"discount": d}, verbose=False)
        m_i.build_model()
        costs_i.append(m_i.solve()["objective_value"])
        
        # Run Model C
        m_c = EnergyCommunityModel(com_data, mode="C", config={"discount": d}, verbose=False)
        m_c.build_model()
        costs_c.append(m_c.solve()["objective_value"])
        
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(discounts, costs_i, marker='o', label="Model I (Individual)")
    plt.plot(discounts, costs_c, marker='s', label="Model C (Community)")
    plt.xlabel("Tariff Discount Ratio (\u03b2)")
    plt.ylabel("Total Daily Cost (DKK)")
    plt.title("Sensitivity: Tariff Discount vs Total Cost")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(out_dir / "Sensitivity_Discount.png", dpi=150)
    plt.close()


def run_penalty_sensitivity(com_data, out_dir: Path):
    """Analyze how changing the penalty price (alpha) affects total costs."""
    print("Running Penalty Sensitivity...")
    penalties = [0, 5, 10, 20, 50]  # DKK/kWh
    costs_i, costs_c = [], []
    
    for p in penalties:
        m_i = EnergyCommunityModel(com_data, mode="I", config={"penalty": p}, verbose=False)
        m_i.build_model()
        costs_i.append(m_i.solve()["objective_value"])
        
        m_c = EnergyCommunityModel(com_data, mode="C", config={"penalty": p}, verbose=False)
        m_c.build_model()
        costs_c.append(m_c.solve()["objective_value"])
        
    plt.figure(figsize=(8, 5))
    plt.plot(penalties, costs_i, marker='o', label="Model I (Individual)")
    plt.plot(penalties, costs_c, marker='s', label="Model C (Community)")
    plt.xlabel("Penalty Price (\u03b1) [DKK/kWh]")
    plt.ylabel("Total Daily Cost (DKK)")
    plt.title("Sensitivity: Penalty Price vs Total Cost")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(out_dir / "Sensitivity_Penalty.png", dpi=150)
    plt.close()


def run_cap_sensitivity(com_data, out_dir: Path):
    """Analyze how changing the capacity limit tightness affects total costs."""
    print("Running Cap Limit Sensitivity...")
    # 1.0 is the baseline. 0.5 is half the capacity (tighter), 1.5 is loose.
    cap_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
    costs_i, costs_c = [], []
    
    for c in cap_factors:
        m_i = EnergyCommunityModel(com_data, mode="I", config={"cap_factor": c}, verbose=False)
        m_i.build_model()
        costs_i.append(m_i.solve()["objective_value"])
        
        m_c = EnergyCommunityModel(com_data, mode="C", config={"cap_factor": c}, verbose=False)
        m_c.build_model()
        costs_c.append(m_c.solve()["objective_value"])
        
    plt.figure(figsize=(8, 5))
    plt.plot(cap_factors, costs_i, marker='o', label="Model I (Individual)")
    plt.plot(cap_factors, costs_c, marker='s', label="Model C (Community)")
    plt.xlabel("Cap Factor (1.0 = Default Contract Limit)")
    plt.ylabel("Total Daily Cost (DKK)")
    plt.title("Sensitivity: Capacity Limit vs Total Cost")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(out_dir / "Sensitivity_Cap.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    # Setup data and output directory
    com = load_community("base", day="d1")
    out_dir = RESULTS_DIR / "base_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run all three experiments
    run_discount_sensitivity(com, out_dir)
    run_penalty_sensitivity(com, out_dir)
    run_cap_sensitivity(com, out_dir)
    
    print(f"Sensitivity analysis complete. Check the '{out_dir.relative_to(Path.cwd())}' folder for plots.")