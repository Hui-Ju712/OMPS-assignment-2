"""Run Assignment 2 experiments from one entry point.

    python main.py                                  # Q1 base comparison (R, I, C)
    python main.py --sensitivity                    # Q1 plus contract sensitivity studies
    python main.py --question 2                     # Q2 pricing model, cap-only variant
    python main.py --question 2 --variant bb        # Q2 with budget balance
    python main.py --question 2 --variant ir        # Q2 with budget balance and individual rationality
    python main.py --community my_mix --day d4      # select community and day
    python main.py --members                        # also print all members' summaries
    python main.py --demo                           # also run the Q3 scenario demonstration
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gurobipy as gp

from src.data_loader import load_community, list_communities
from src.model import EnergyCommunityModel
from src.plotting import plot_community_schedule, plot_consumer_cost_comparison
from src.sensitivity import run_cap_sensitivity, run_discount_sensitivity, run_penalty_sensitivity

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
    plot_consumer_cost_comparison(
        cost_results,
        title="Base Case: Consumer Cost Comparison",
        save_to=results_dir / "Q1_consumer_cost_comparison.png",
    )
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
    

def _solve_model(data, mode, config, verbose):
    model = EnergyCommunityModel(data, mode=mode, config=config, verbose=verbose)
    model.build_model()
    return model, model.solve()


def run_q2_pricing(data, args, results_dir=RESULTS_DIR):
    """Run the selected pricing variant with the same community data as Q1."""
    common = {"cap_factor": args.cap_factor, "daily_energy_rule": args.daily_energy_rule}
    if args.discount is not None:
        common["discount"] = args.discount
    if args.penalty is not None:
        common["penalty"] = args.penalty
    config = dict(common, budget_balance=args.variant in {"bb", "ir"},
                  individual_rationality=args.variant == "ir",
                  price_bounds=(args.price_min, args.price_max),
                  time_limit=args.time_limit, mip_gap=args.mip_gap)
    references = {}
    # IR outside options are recalculated under the same contract and daily-energy rule.
    if args.variant == "ir":
        modes = ("R", "I") if args.ir_reference == "both" else (args.ir_reference,)
        for mode in modes:
            _, references[mode] = _solve_model(data, mode, common, args.verbose)
        config["benchmark_costs"] = {
            member.consumer_id: min(result["consumer_costs"][member.consumer_id]
                                    for result in references.values())
            for member in data.members
        }
    model = EnergyCommunityModel(data, mode="DP", config=config, verbose=args.verbose)
    model.build_model()
    if args.export_model:
        args.export_model.parent.mkdir(parents=True, exist_ok=True)
        model.m.write(str(args.export_model))
        print(f"Exported {model.m.NumVars} variables, {model.m.NumConstrs} rows, "
              f"{model.m.NumSOS} SOS1 constraints to {args.export_model}")
        return
    _, central = _solve_model(data, "C", common, args.verbose)
    result = model.solve()
    output = args.output_dir or results_dir / args.community / f"Q2_{args.day}_{args.variant}"
    output.mkdir(parents=True, exist_ok=True)
    report = {"configuration": config, "day": args.day,
              "solution_concept": "optimistic", "DP": result, "C": central,
              "outside_options": references,
              "cost_above_C_DKK": result["objective_value"] - central["objective_value"]}
    if "benchmark_costs" in config:
        report["individual_gains_DKK"] = {
            cid: benchmark - result["consumer_costs"][cid]
            for cid, benchmark in config["benchmark_costs"].items()
        }
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "schedule.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["hour", "internal_price_DKK_per_kWh", "import_kW", "export_kW",
                         "cap_kW", "exceedance_kW"])
        for t in model.T:
            writer.writerow([t, result["internal_prices"][t], model.var["P_imp"][t].X,
                             model.var["P_exp"][t].X, model.cap_limit[t],
                             model.var["P_exc"][t].X])
    plot_consumer_cost_comparison(
        {"C": central, "DP": result},
        models=("C", "DP"),
        title=f"{args.community} {args.day} - Q2 {args.variant.upper()}: C vs DP Consumer Costs",
        save_to=output / "consumer_cost_comparison.png",
    )
    print(f"DP: {result['objective_value']:.6f} DKK ({result['status']}, "
          f"gap {result['mip_gap']:.3g}); C: {central['objective_value']:.6f} DKK")
    print(f"Manager balance: {result['manager_balance']:.6f} DKK")
    print(f"Results: {output}")

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base", help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration")
    ap.add_argument("--sensitivity", action="store_true", help="run tariff, penalty, and capacity sensitivity analyses")
    ap.add_argument("--question", type=int, choices=(1, 2), default=1,
                    help="experiment to run (default: 1, the existing Q1 comparison)")
    q2 = ap.add_argument_group("Question 2 options (use with --question 2)")
    q2.add_argument("--variant", choices=("cap", "bb", "ir"), default="cap")
    q2.add_argument("--ir-reference", choices=("R", "I", "both"), default="both")
    q2.add_argument("--discount", type=float)
    q2.add_argument("--penalty", type=float)
    q2.add_argument("--cap-factor", type=float, default=1.0)
    q2.add_argument("--price-min", type=float)
    q2.add_argument("--price-max", type=float)
    q2.add_argument("--daily-energy-rule", choices=("fixed", "minimum"), default="fixed")
    q2.add_argument("--time-limit", type=float, default=300.0)
    q2.add_argument("--mip-gap", type=float, default=1e-4)
    q2.add_argument("--export-model", type=Path, help="write the DP LP/SOS model without solving")
    q2.add_argument("--output-dir", type=Path)
    q2.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.question == 2 and args.sensitivity:
        ap.error("--sensitivity runs the Q1 studies; use --question 1 (the default)")
    if args.question == 1 and args.export_model:
        ap.error("--export-model requires --question 2")

    com = load_community(args.community, day=args.day)
    print(com.summary(), "\n")
    if args.members:
        for m in com.members:
            print(m.summary(), "\n")
    print("Cap profile (kW):", " ".join(f"{x:.1f}" for x in com.contract.capacity_limit_kW))
    ic = com.contract.individual_capacity_limit_kW
    print("Individual caps in model I (kW), hour 0 / hour 11:",
          "  ".join(f"{cid} ({com.member(cid).archetype}) {ic[cid][0]:.2f}/{ic[cid][11]:.2f}" for cid in ("C1", "C1_PVB", "C4", "C4_PVB") if cid in com.member_ids))
    if args.question == 1:
        out_dir = RESULTS_DIR / args.community
        out_dir.mkdir(parents=True, exist_ok=True)
        run_q1_base_comparison(com, out_dir)
    else:
        run_q2_pricing(com, args, RESULTS_DIR)
    if args.sensitivity:
        sensitivity_dir = RESULTS_DIR / "base_sensitivity"
        sensitivity_dir.mkdir(parents=True, exist_ok=True)
        run_discount_sensitivity(com, sensitivity_dir)
        run_penalty_sensitivity(com, sensitivity_dir)
        run_cap_sensitivity(com, sensitivity_dir)
        print(f"Sensitivity analysis complete. Check the '{sensitivity_dir.relative_to(Path.cwd())}' folder for plots.")
    if args.demo:
        import runpy
        runpy.run_module("src.scenario_generation", run_name="__main__")


if __name__ == "__main__":
    try:
        main()
    except gp.GurobiError as error:
        if error.errno == 10010:
            raise SystemExit(
                "The model exceeds the restricted Gurobi licence. Activate an academic licence. "
                "To inspect Q2 without solving, run: python main.py --question 2 "
                "--variant cap --export-model results/Q2_DP.lp"
            ) from error
        raise
