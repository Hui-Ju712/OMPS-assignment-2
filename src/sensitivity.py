"""Contract sensitivity analysis for community value and DSO congestion."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import CommunityData
from src.model import EnergyCommunityModel
from src.plotting import plot_sensitivity

REFERENCE_COST_TOLERANCE_DKK = 0.02
BASE_VALUE_TOLERANCE_DKK = 0.1
COMPARISON_TOLERANCE_DKK = 1e-6
MODEL_LABELS = {
    "R": "Model R (Reference)",
    "I": "Model I (Individual)",
    "C": "Model C (Community)",
}


def _solve_model(
    com_data: CommunityData,
    mode: str,
    config: dict[str, float] | None = None,
) -> tuple[EnergyCommunityModel, dict]:
    """Build and solve one model for the supplied mode and contract terms."""
    model = EnergyCommunityModel(com_data, mode=mode, config=config, verbose=False)
    model.build_model()
    return model, model.solve()


def _hourly_imports(model: EnergyCommunityModel) -> list[float]:
    """Return the connection-point import profile under the common comparison rule."""
    if model.mode == "C":
        return [model.var["P_imp"][t].X for t in model.T]
    return [
        sum(model.var["q_imp"][i, t].X for i in range(len(model.members)))
        for t in model.T
    ]


def _total_exceedance(imports: list[float], cap: np.ndarray) -> float:
    """Return daily energy imported above the given hourly cap, in kWh."""
    return float(sum(max(imported - limit, 0.0) for imported, limit in zip(imports, cap)))


def _validate_reference(com_data: CommunityData, reference_cost: float, reference_imports: list[float]) -> None:
    """Check the starter package's reference cost and congestion hours for base d1."""
    if com_data.community != "base" or com_data.day != "d1":
        return

    starter_cost = com_data.contract.reference_model_R.get("cost_DKK")
    if starter_cost is not None and abs(reference_cost - starter_cost) > REFERENCE_COST_TOLERANCE_DKK:
        raise RuntimeError(
            f"Base-case Model R cost {reference_cost:.6f} DKK does not match "
            f"the starter value {starter_cost:.2f} DKK."
        )

    cap = com_data.contract.capacity_limit_kW
    above_cap_hours = sum(imported > limit + COMPARISON_TOLERANCE_DKK for imported, limit in zip(reference_imports, cap))
    if above_cap_hours != 9:
        raise RuntimeError(f"Expected base-case Model R to exceed the cap in 9 hours, got {above_cap_hours}.")
    print(f"Verified Model R base case: {reference_cost:.2f} DKK and {above_cap_hours} hours above cap.")


def _evaluate_sweep(
    com_data: CommunityData,
    parameter: str,
    parameter_values: list[float],
    config_key: str,
    baseline_config: dict[str, float],
    reference_cost: float,
    reference_imports: list[float],
) -> tuple[dict[str, list[float]], dict[str, list[float]], list[dict[str, float | str]]]:
    """Solve I and C across a sweep; return community value, congestion, and CSV rows."""
    values = {label: [] for label in MODEL_LABELS.values()}
    exceedances = {label: [] for label in MODEL_LABELS.values()}
    rows: list[dict[str, float | str]] = []

    for parameter_value in parameter_values:
        config = {**baseline_config, config_key: parameter_value}
        cap = com_data.contract.capacity_limit_kW * config["cap_factor"]
        r_exceedance = _total_exceedance(reference_imports, cap)
        values[MODEL_LABELS["R"]].append(0.0)
        exceedances[MODEL_LABELS["R"]].append(r_exceedance)
        rows.append({
            "parameter": parameter,
            "parameter_value": parameter_value,
            "model": MODEL_LABELS["R"],
            "objective_value_DKK": reference_cost,
            "value_vs_R_DKK": 0.0,
            "import_exceedance_kWh": r_exceedance,
        })

        costs: dict[str, float] = {}
        for mode in ("I", "C"):
            model, result = _solve_model(com_data, mode, config)
            cost = float(result["objective_value"])
            excess = _total_exceedance(_hourly_imports(model), cap)
            costs[mode] = cost
            values[MODEL_LABELS[mode]].append(reference_cost - cost)
            exceedances[MODEL_LABELS[mode]].append(excess)
            rows.append({
                "parameter": parameter,
                "parameter_value": parameter_value,
                "model": MODEL_LABELS[mode],
                "objective_value_DKK": cost,
                "value_vs_R_DKK": reference_cost - cost,
                "import_exceedance_kWh": excess,
            })

        if costs["C"] > costs["I"] + COMPARISON_TOLERANCE_DKK:
            raise RuntimeError(
                f"Model C costs more than Model I at {parameter}={parameter_value:g}: "
                f"{costs['C']:.6f} > {costs['I']:.6f} DKK."
            )

    return values, exceedances, rows


def _check_base_values(
    com_data: CommunityData,
    rows: list[dict[str, float | str]],
    baseline_parameter: float,
    parameter: str,
) -> None:
    """Check the requested I/C community-value results at the base contract terms."""
    if com_data.community != "base" or com_data.day != "d1":
        return
    baseline_rows = [
        row for row in rows
        if row["parameter"] == parameter
        and abs(float(row["parameter_value"]) - baseline_parameter) < 1e-9
    ]
    results = {str(row["model"]): float(row["value_vs_R_DKK"]) for row in baseline_rows}
    expected = {
        MODEL_LABELS["I"]: -81.89170709087108,
        MODEL_LABELS["C"]: 26.07925970914207,
    }
    if len(baseline_rows) == 0:
        raise RuntimeError(f"The {parameter} sweep does not include its base-case value {baseline_parameter:g}.")
    for model, expected_value in expected.items():
        if model not in results or abs(results[model] - expected_value) > BASE_VALUE_TOLERANCE_DKK:
            raise RuntimeError(
                f"Base-case {model} value against R is {results.get(model)!r} DKK; "
                f"expected about {expected_value:+.2f} DKK."
            )
    print(
        f"Verified base {parameter}: I value {results[MODEL_LABELS['I']]:+.2f} DKK, "
        f"C value {results[MODEL_LABELS['C']]:+.2f} DKK."
    )


def _print_interesting_terms(
    parameter: str,
    parameter_values: list[float],
    values: dict[str, list[float]],
    exceedances: dict[str, list[float]],
) -> None:
    """Print sampled contract terms where community value is positive and congestion falls."""
    print(f"\n{parameter}: terms where the community gains and DSO exceedance is below R:")
    found = False
    for mode in ("I", "C"):
        label = MODEL_LABELS[mode]
        terms = [
            f"{value:g}"
            for value, gain, excess, r_excess in zip(
                parameter_values,
                values[label],
                exceedances[label],
                exceedances[MODEL_LABELS["R"]],
            )
            if gain > COMPARISON_TOLERANCE_DKK and excess < r_excess - COMPARISON_TOLERANCE_DKK
        ]
        if terms:
            found = True
            print(f"  {mode}: {', '.join(terms)}")
    if not found:
        print("  None at the sampled parameter values.")


def run_sensitivity_study(
    com_data: CommunityData,
    out_dir: Path,
    reference_model: EnergyCommunityModel | None = None,
    reference_result: dict | None = None,
) -> None:
    """Compare I and C with one shared R reference over beta, alpha, and cap factor."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if (reference_model is None) != (reference_result is None):
        raise ValueError("Provide both reference_model and reference_result, or neither.")
    if reference_model is None:
        print("Solving Model R reference...")
        reference_model, reference_result = _solve_model(com_data, "R")
    else:
        print("Reusing the base-case Model R reference...")
    if reference_result is None:
        raise RuntimeError("Model R did not produce a reference result.")
    reference_cost = float(reference_result["objective_value"])
    reference_imports = _hourly_imports(reference_model)
    _validate_reference(com_data, reference_cost, reference_imports)

    contract = com_data.contract
    baseline_config = {
        "discount": contract.tariff_discount_ratio,
        "penalty": contract.penalty_DKK_per_kWh,
        "cap_factor": 1.0,
    }
    maximum_cap_factor = max(
        imported / cap
        for imported, cap in zip(reference_imports, contract.capacity_limit_kW)
        if cap > 0
    )

    discount_values = sorted(set(np.linspace(0.0, 1.0, 21).tolist() + [baseline_config["discount"]]))
    penalty_minimum = (1.0 - baseline_config["discount"]) * com_data.import_tariff
    penalty_values = sorted({
        *(
            penalty_minimum * factor
            for factor in (1, 2, 4, 8, 16, 32, 64)
            if penalty_minimum * factor <= baseline_config["penalty"]
        ),
        baseline_config["penalty"],
    })
    cap_values = sorted(set(
        np.linspace(0.25, maximum_cap_factor, 12).tolist()
        + [1.0]
    ))

    sweeps = [
        ("discount", "discount", discount_values, "Tariff Discount Ratio (β)", "Sensitivity: Community Value and DSO Congestion vs Discount", "Sensitivity_Discount.png", baseline_config["discount"]),
        ("penalty", "penalty", penalty_values, "Penalty Price (α) [DKK/kWh]", "Sensitivity: Community Value and DSO Congestion vs Penalty", "Sensitivity_Penalty.png", baseline_config["penalty"]),
        ("cap_factor", "cap_factor", cap_values, "Cap Factor (f)", "Sensitivity: Community Value and DSO Congestion vs Cap Factor", "Sensitivity_Cap.png", 1.0),
    ]

    all_rows: list[dict[str, float | str]] = []
    for parameter, config_key, parameter_values, xlabel, title, filename, baseline_value in sweeps:
        print(f"Running {parameter} sensitivity...")
        values, exceedances, rows = _evaluate_sweep(
            com_data,
            parameter,
            parameter_values,
            config_key,
            baseline_config,
            reference_cost,
            reference_imports,
        )
        _check_base_values(com_data, rows, baseline_value, parameter)
        figure = plot_sensitivity(
            parameter_values,
            values,
            exceedances,
            xlabel=xlabel,
            title=title,
            save_to=out_dir / filename,
        )
        plt.close(figure)
        all_rows.extend(rows)
        _print_interesting_terms(parameter, parameter_values, values, exceedances)

    csv_path = out_dir / "Sensitivity_Results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "parameter",
            "parameter_value",
            "model",
            "objective_value_DKK",
            "value_vs_R_DKK",
            "import_exceedance_kWh",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nModel R base cost: {reference_cost:.2f} DKK")
    print(f"Reference peak import: {max(reference_imports):.2f} kW")
    print(f"Cap factor where R is entirely below the cap: {maximum_cap_factor:.3f}")
    print(f"Sensitivity values saved to {csv_path}")
