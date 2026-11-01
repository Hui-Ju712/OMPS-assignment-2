"""Entry point of the Assignment 2 starter package: load a community and summarize it.

    python main.py                                   # base case (16 members), day d1
    python main.py --community my_mix                # your own data/params_community_my_mix.json
    python main.py --day d4                          # another day of the week (and that day's cap)
    python main.py --members                         # also print every member's Assignment 1 summary
    python main.py --demo                            # also run the scenario-generation demonstration (Question 3)

No model is provided: build your Assignment 1 model class on the ``InputData`` of each member, then the community
models of the assignment (R, I, C, DP, ...) on top of it, one function per experiment, writing to ``results/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_community, list_communities
from src.model import EnergyCommunityModel
from src.plotting import plot_community_schedule

RESULTS_DIR = Path(__file__).resolve().parent / "results"

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
    
    # Print the cost comparison
    print("\n--- Results Summary (Total Cost) ---")
    print(f"Model R: {res_r['objective_value']:.2f} DKK")
    print(f"Model I: {res_i['objective_value']:.2f} DKK")
    print(f"Model C: {res_c['objective_value']:.2f} DKK")
    
    # Display the extracted dual variables (internal prices) from Model C
    print(f"\nModel C Internal Prices (first 5 hours):")
    print([round(p, 3) for p in res_c['internal_prices'][:5]])

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base", help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration")
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
    run_q1_base_comparison(com, out_dir)
    if args.demo:
        import runpy
        runpy.run_module("src.scenario_generation", run_name="__main__")


if __name__ == "__main__":
    main()
