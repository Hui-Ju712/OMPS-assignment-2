"""Run Assignment 2 experiments from one entry point.

    python main.py                                  # Q1 base comparison (R, I, C)
    python main.py --sensitivity                    # Q1 plus contract sensitivity studies
    python main.py --question 2                     # Q2 pricing model, cap-only variant
    python main.py --question 2 --variant all       # all three Q2 variants, saved as CSV
    python main.py --question 2 --variant bb        # Q2 with budget balance
    python main.py --question 2 --variant ir        # Q2 with budget balance and individual rationality
    python main.py --community my_mix --day d4      # select community and day
    python main.py --members                        # also print all members' summaries
    python main.py --demo                           # also run the Q3 scenario demonstration
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gurobipy as gp

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

def _solve_model(data, mode, config, verbose):
    model = EnergyCommunityModel(data, mode=mode, config=config, verbose=verbose)
    model.build_model()
    return model, model.solve()


def save_q2_results(models, results, output, args):
    """Save actual solver results as four CSVs; never read precomputed results."""
    output.mkdir(parents=True, exist_ok=True)
    reference_r = results["R"]["consumer_costs"]
    reference_i = results["I"]["consumer_costs"]
    central_cost = results["C"]["objective_value"]
    summaries, member_costs, schedules, member_schedules = [], [], [], []

    for label, result in results.items():
        model = models[label]
        solved = result.get("objective_value") is not None
        summary = {
            "model": label, "status": result["status"],
            "solver_status": result.get("solver_status"), "solver": "Gurobi",
            "cost_DKK": result.get("objective_value"),
            "objective_bound_DKK": result.get("objective_bound",
                                              result.get("objective_value")),
            "mip_gap": result.get("mip_gap"),
            "runtime_seconds": result.get("runtime_seconds"),
            "external_grid_cost_DKK": result.get("external_grid_cost"),
            "manager_balance_DKK": result.get("manager_balance"),
            "cost_above_C_DKK": result["objective_value"] - central_cost if solved else None,
            "saving_against_R_DKK": results["R"]["objective_value"] - result["objective_value"] if solved else None,
            "peak_import_kW": None, "daily_import_kWh": None, "daily_export_kWh": None,
            "cap_exceedance_kWh": None, "max_cap_exceedance_kW": None,
            "hours_above_cap": None, "members_worse_off_vs_R": None,
            "members_worse_off_vs_I": None, "members_worse_off_vs_best": None,
            "worst_gain_vs_best_DKK": None,
            "kkt_primal_residual": result.get("kkt_primal_residual"),
            "kkt_stationarity_residual": result.get("kkt_stationarity_residual"),
            "kkt_complementarity_residual": result.get("kkt_complementarity_residual"),
            "payment_identity_residual": result.get("payment_identity_residual"),
            "system_cost_check_DKK": result.get("system_cost_check"),
            "day": args.day, "daily_energy_rule": args.daily_energy_rule,
            "price_scope": model.config.get("price_scope", "uniform"),
            "discount": model.discount, "penalty_DKK_per_kWh": model.penalty,
            "cap_factor": model.cap_factor, "IR_reference": args.ir_reference,
            "price_min": args.price_min, "price_max": args.price_max,
            "time_limit_seconds": args.time_limit, "target_mip_gap": args.mip_gap,
        }
        if solved:
            imports, exports = [], []
            for t in model.T:
                if model.mode in {"R", "I"}:
                    imp = sum(model.var["q_imp"][i, t].X for i in range(len(model.members)))
                    exp = sum(model.var["q_exp"][i, t].X for i in range(len(model.members)))
                else:
                    imp, exp = model.var["P_imp"][t].X, model.var["P_exp"][t].X
                imports.append(imp)
                exports.append(exp)
                schedules.append({
                    "model": label, "hour": t,
                    "internal_price_DKK_per_kWh": result.get("internal_prices", [None] * 24)[t],
                    "import_kW": imp, "export_kW": exp,
                    "cap_kW": float(model.cap_limit[t]),
                    "exceedance_kW": max(0, imp - model.cap_limit[t]),
                })
            exceedance = [max(0, imp-model.cap_limit[t]) for t, imp in enumerate(imports)]
            gains_r, gains_i, gains_best = [], [], []
            for i, member in enumerate(model.members):
                cid = member.consumer_id
                cost = result["consumer_costs"][cid]
                gain_r, gain_i = reference_r[cid]-cost, reference_i[cid]-cost
                gain_best = min(reference_r[cid], reference_i[cid])-cost
                gains_r.append(gain_r)
                gains_i.append(gain_i)
                gains_best.append(gain_best)
                member_costs.append({
                    "model": label, "consumer_id": cid, "archetype": member.archetype,
                    "cost_DKK": cost, "R_cost_DKK": reference_r[cid],
                    "I_cost_DKK": reference_i[cid], "gain_vs_R_DKK": gain_r,
                    "gain_vs_I_DKK": gain_i, "gain_vs_best_DKK": gain_best,
                    "internal_payment_DKK": result.get("member_payments", {}).get(cid),
                })
                for t in model.T:
                    row = {"model": label, "consumer_id": cid,
                           "archetype": member.archetype, "hour": t}
                    for key in ("load", "pv", "p_ch", "p_dis", "soc", "dev"):
                        row[key] = model.var[key][i, t].X
                    row["net_consumption_kWh"] = row["load"] + row["p_ch"] - row["pv"] - row["p_dis"]
                    prices = result.get("internal_prices_by_member", {}).get(
                        cid, result.get("internal_prices", [None] * 24)
                    )
                    row["internal_price_DKK_per_kWh"] = prices[t]
                    member_schedules.append(row)
            summary.update(
                peak_import_kW=max(imports), daily_import_kWh=sum(imports),
                daily_export_kWh=sum(exports), cap_exceedance_kWh=sum(exceedance),
                max_cap_exceedance_kW=max(exceedance),
                hours_above_cap=sum(value > 1e-6 for value in exceedance),
                members_worse_off_vs_R=sum(value < -1e-6 for value in gains_r),
                members_worse_off_vs_I=sum(value < -1e-6 for value in gains_i),
                members_worse_off_vs_best=sum(value < -1e-6 for value in gains_best),
                worst_gain_vs_best_DKK=min(gains_best),
            )
        summaries.append(summary)

    for filename, rows in (("summary.csv", summaries), ("member_costs.csv", member_costs),
                           ("schedule.csv", schedules), ("members_schedule.csv", member_schedules)):
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def q2_output_directory(args, results_dir=RESULTS_DIR):
    scope = "individual" if args.variant == "ir" else (args.price_scope or "uniform")
    suffix = "_individual" if scope == "individual" else ""
    return args.output_dir or results_dir / args.community / f"Q2_{args.day}_{args.variant}{suffix}"


def run_q2_pricing(data, args, results_dir=RESULTS_DIR):
    """Solve Q2 and save CSVs directly from the returned results and model variables."""
    common = {"cap_factor": args.cap_factor, "daily_energy_rule": args.daily_energy_rule}
    if args.discount is not None:
        common["discount"] = args.discount
    if args.penalty is not None:
        common["penalty"] = args.penalty
    variants = ("cap", "bb", "ir") if args.variant == "all" else (args.variant,)
    models, results = {}, {}

    def configuration(variant):
        config = dict(common, budget_balance=variant in {"bb", "ir"},
                      individual_rationality=variant == "ir",
                      price_bounds=(args.price_min, args.price_max),
                      price_scope="individual" if variant == "ir" else (args.price_scope or "uniform"),
                      time_limit=args.time_limit, mip_gap=args.mip_gap)
        if variant == "ir":
            references = ("R", "I") if args.ir_reference == "both" else (args.ir_reference,)
            config["benchmark_costs"] = {
                member.consumer_id: min(results[mode]["consumer_costs"][member.consumer_id]
                                        for mode in references)
                for member in data.members
            }
        return config

    # Cap/BB export needs no licence-sized optimisation or result files.
    if args.export_model and args.variant != "ir":
        model = EnergyCommunityModel(data, mode="DP", config=configuration(args.variant),
                                     verbose=args.verbose)
        model.build_model()
        args.export_model.parent.mkdir(parents=True, exist_ok=True)
        model.m.write(str(args.export_model))
        print(f"Exported model to {args.export_model}")
        return

    for mode in ("R", "I", "C"):
        models[mode], results[mode] = _solve_model(data, mode, common, args.verbose)
    output = q2_output_directory(args, results_dir)
    if not args.export_model:
        save_q2_results(models, results, output, args)
    for variant in variants:
        label = "DP_ir_individual" if variant == "ir" else f"DP_{variant}"
        model = EnergyCommunityModel(data, mode="DP", config=configuration(variant),
                                     verbose=args.verbose)
        model.build_model()
        if args.export_model:
            args.export_model.parent.mkdir(parents=True, exist_ok=True)
            model.m.write(str(args.export_model))
            print(f"Exported model to {args.export_model}")
            return
        models[label] = model
        if model.config.get("price_scope") == "individual":
            supported = model.warm_start_from_central(models["C"])
            print(f"{label}: central-dispatch pricing start {'found' if supported else 'not found'}")
        results[label] = model.solve()
        # Checkpoint each solve, including infeasible/time-limit runs with no incumbent.
        save_q2_results(models, results, output, args)
        cost = results[label].get("objective_value")
        cost_text = f"{cost:.6f} DKK" if cost is not None else "no feasible solution found"
        print(f"{label}: {cost_text} ({results[label]['status']})")
    print(f"CSV results saved by save_q2_results(): {output}")
    from src.q2_reporting import generate_q2_report
    generate_q2_report(output, args.compare_results, args.diagnose_ir, args.community)

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
    q2.add_argument("--variant", choices=("cap", "bb", "ir", "all"), default="cap")
    q2.add_argument("--price-scope", choices=("uniform", "individual"), default=None,
                    help="cap/BB pricing scope (default: uniform); IR always uses individual prices")
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
    q2.add_argument("--report-only", action="store_true",
                    help="create plots and tables from saved CSVs without rerunning DP")
    q2.add_argument("--compare-results", type=Path, action="append", default=[],
                    help="also compare an existing Q2 result folder; repeat for several folders")
    q2.add_argument("--diagnose-ir", action="store_true",
                    help="minimize participation shortfalls by repricing the saved fixed BB/C dispatch")
    args = ap.parse_args(argv)
    if args.variant == "ir" and args.price_scope == "uniform":
        ap.error("The IR variant now uses individual prices; omit --price-scope or use individual")
    if args.report_only:
        if args.question != 2 or args.export_model:
            ap.error("--report-only requires --question 2 and cannot be combined with --export-model")
        from src.q2_reporting import generate_q2_report
        generate_q2_report(q2_output_directory(args), args.compare_results,
                           args.diagnose_ir, args.community)
        return
    if args.variant == "all" and args.export_model:
        ap.error("--export-model needs a single variant: cap, bb or ir")
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
    out_dir = RESULTS_DIR / args.community
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.question == 1:
        model_r, res_r = run_q1_base_comparison(com, out_dir)
    else:
        run_q2_pricing(com, args, RESULTS_DIR)

    if args.sensitivity:
        sensitivity_dir = RESULTS_DIR / "base_sensitivity"
        sensitivity_dir.mkdir(parents=True, exist_ok=True)
        run_sensitivity_study(com, sensitivity_dir, reference_model=model_r, reference_result=res_r)
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
