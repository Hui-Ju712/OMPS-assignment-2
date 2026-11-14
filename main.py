"""Entry point of the Assignment 2 starter package: load a community and summarize it.

    python main.py                                   # base case (16 members), day d1
    python main.py --sensitivity                     # run the sensitivity analyses (discount, penalty, capacity)
    python main.py --community my_mix                # your own data/params_community_my_mix.json
    python main.py --day d4                          # another day of the week (and that day's cap)
    python main.py --members                         # also print every member's Assignment 1 summary
    python main.py --demo                            # also run the scenario-generation demonstration (Question 3)

No model is provided: build your Assignment 1 model class on the ``InputData`` of each member, then the community
models of the assignment (R, I, C, DP, ...) on top of it, one function per experiment, writing to ``results/``.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.data_loader import load_community, list_communities
from src.model import EnergyCommunityModel
from src.plotting import plot_community_schedule
from src.sensitivity import run_sensitivity_study

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def write_cost_breakdown_csv(results_by_mode, output_path):
    """Write consumer costs and model-level settlement values to a CSV file."""
    fieldnames = [
        "model",
        "row_type",
        "consumer_id",
        "consumer_cost_DKK",
        "objective_value_DKK",
        "external_grid_cost_DKK",
        "manager_balance_DKK",
        "system_cost_check_DKK",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for mode, result in results_by_mode.items():
            for consumer_id, cost in result["consumer_costs"].items():
                writer.writerow({
                    "model": mode,
                    "row_type": "CONSUMER",
                    "consumer_id": consumer_id,
                    "consumer_cost_DKK": f"{cost:.6f}",
                })
            writer.writerow({
                "model": mode,
                "row_type": "SUMMARY",
                "objective_value_DKK": f"{result['objective_value']:.6f}",
                "external_grid_cost_DKK": f"{result['external_grid_cost']:.6f}",
                "manager_balance_DKK": f"{result['manager_balance']:.6f}",
                "system_cost_check_DKK": f"{result['system_cost_check']:.6f}",
            })

def run_q1_base_comparison(com_data, results_dir=RESULTS_DIR):
    """Experiment 1: Compare R, I, and C models on the base case data."""
    print("\n--- Running Question 1: Base Case Comparison ---")
    
    # Model R: No contract
    print("Solving Model R (Reference)...")
    model_r = EnergyCommunityModel(data=com_data, mode="R", verbose=False)
    model_r.build_model()
    res_r = model_r.solve()
    plot_community_schedule(model_r, res_r, title="Base Case", save_to=results_dir / "Q1_Model_R_Schedule.png")
    
    # Model I: Individual contracts
    print("Solving Model I (Individual Contracts)...")
    model_i = EnergyCommunityModel(data=com_data, mode="I", verbose=False)
    model_i.build_model()
    res_i = model_i.solve()
    plot_community_schedule(model_i, res_i, title="Base Case", save_to=results_dir / "Q1_Model_I_Schedule.png")
    
    # Model C: Community contract (Central Dispatch)
    print("Solving Model C (Community Contract)...")
    model_c = EnergyCommunityModel(data=com_data, mode="C", verbose=False)
    model_c.build_model()
    res_c = model_c.solve()
    plot_community_schedule(model_c, res_c, title="Base Case", save_to=results_dir / "Q1_Model_C_Schedule.png")

    cost_results = {"R": res_r, "I": res_i, "C": res_c}
    cost_csv = results_dir / "Q1_cost_breakdown.csv"
    write_cost_breakdown_csv(cost_results, cost_csv)
    print(f"Cost breakdown CSV: {cost_csv}")
    
    # Print the cost comparison
    print("\n--- Results Summary (Total Cost) ---")
    print(f"Model R: {res_r['objective_value']:.2f} DKK")
    print(f"Model I: {res_i['objective_value']:.2f} DKK")
    print(f"Model C: {res_c['objective_value']:.2f} DKK")
    
    # Display the extracted dual variables (internal prices) from Model C
    print("Model C Internal Prices:")
    for hour, price in enumerate(res_c["internal_prices"]):
        print(f"Hour {hour:02d}: {price:.3f} DKK/kWh")
    return model_r, res_r

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base", help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration")
    ap.add_argument("--sensitivity", action="store_true", help="run tariff, penalty, and capacity sensitivity analyses")
    args = ap.parse_args()

    com = load_community(args.community, day=args.day)
    print(com.summary(), "\n")
    if args.members:
        for m in com.members:
            print(m.summary(), "\n")
    print("Cap profile (kW):", " ".join(f"{x:.1f}" for x in com.contract.capacity_limit_kW))
    ic = com.contract.individual_capacity_limit_kW
    print("Individual caps in model I (kW), hour 0 / hour 11:",
          "  ".join(f"{cid} ({com.member(cid).archetype}) {ic[cid][0]:.2f}/{ic[cid][11]:.2f}" for cid in ("C1", "C1_PVB", "C4", "C4_PVB") if cid in com.member_ids))
    (RESULTS_DIR / args.community).mkdir(parents=True, exist_ok=True)

    out_dir = RESULTS_DIR / args.community
    out_dir.mkdir(parents=True, exist_ok=True)

    # Execute the Q1 experiment
    model_r, res_r = run_q1_base_comparison(com, out_dir)
    if args.sensitivity:
        sensitivity_dir = RESULTS_DIR / "base_sensitivity"
        sensitivity_dir.mkdir(parents=True, exist_ok=True)
        run_sensitivity_study(com, sensitivity_dir, reference_model=model_r, reference_result=res_r)
        print(f"Sensitivity analysis complete. Check the '{sensitivity_dir.relative_to(Path.cwd())}' folder for plots.")
    if args.demo:
        import runpy
        runpy.run_module("src.scenario_generation", run_name="__main__")


if __name__ == "__main__":
    main()
