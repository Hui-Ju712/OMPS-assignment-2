from __future__ import annotations

from pathlib import Path
from typing import Any 

import matplotlib.pyplot as plt
import numpy as np

def _finish(fig: plt.Figure, save_to: Path | str | None) -> plt.Figure:
    """Finalize and optionally save a figure."""
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig

def plot_community_schedule(
    model_obj: Any, 
    results_dict: dict, 
    title: str = "Community Schedule", 
    save_to: Path | str | None = None
) -> plt.Figure:
    """Plot the 24-hour aggregate community schedule for Model R, I, or C.

    This visualizes the total community load, PV utilization, net grid exchange,
    the DSO capacity limit, and both external and internal pricing signals.

    Args:
        model_obj: The formulated EnergyCommunityModel instance.
        results_dict: The dictionary returned by model_obj.solve().
        title: Title prefix for the plot.
        save_to: Optional image path; the path has no physical units.

    Returns:
        Matplotlib figure comparing energy flows against the capacity limit and prices.
    """
    fig, ax1 = plt.subplots(figsize=(12, 5))
    T = list(range(24))
    
    # 1. Extract aggregate community data based on the active model mode
    if model_obj.mode in ["R", "I"]:
        total_imp = [sum(model_obj.var["q_imp"][i, t].X for i in range(len(model_obj.members))) for t in T]
        total_exp = [sum(model_obj.var["q_exp"][i, t].X for i in range(len(model_obj.members))) for t in T]
    else:  # Model C
        total_imp = [model_obj.var["P_imp"][t].X for t in T]
        total_exp = [model_obj.var["P_exp"][t].X for t in T]
        
    total_load = [sum(model_obj.var["load"][i, t].X for i in range(len(model_obj.members))) for t in T]
    total_pv = [sum(model_obj.var["pv"][i, t].X for i in range(len(model_obj.members))) for t in T]
    
    # Calculate net grid exchange (positive for import, negative for export)
    net_grid = [total_imp[t] - total_exp[t] for t in T]
    cap = model_obj.cap_limit
    
    # 2. Left Y-axis: Energy (kWh/h) - Plotting physical energy flows
    ax1.bar(T, total_load, label="Total Load", color="tab:blue", alpha=0.7)
    ax1.bar(T, [-p for p in total_pv], label="Total PV Used (negative = generation)", color="orange", alpha=0.7)
    
    ax1.plot(T, net_grid, marker='o', markersize=4, color='black', label='Net import (+) / export (-)')
    ax1.step(T, cap, where='mid', color='red', linestyle='--', linewidth=1.5, label='DSO Cap (Limit)')
    
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Energy (kWh/h)")
    ax1.axhline(0, color='gray', linewidth=1)
    
    # 3. Right Y-axis: Price (DKK/kWh) - Plotting economic signals
    ax2 = ax1.twinx()
    prices = model_obj.data.energy_price
    
    # Day-ahead energy price (solid red step line)
    ax2.step(T, prices, where='mid', color='tab:red', label='Day-ahead Price')
    
    # Internal clearing price (only applicable to Model C, dashed purple line)
    if model_obj.mode == "C" and "internal_prices" in results_dict:
        ax2.step(T, results_dict["internal_prices"], where='mid', color='purple', linestyle='--', label=r'Internal Price ($\lambda$)')
        
    ax2.set_ylabel("Price (DKK/kWh)")
    
    # 4. Formatting legend and layout
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    fig.suptitle(f"{title} - Mode {model_obj.mode} (Cost: {results_dict['objective_value']:.2f} DKK)")
    
    return _finish(fig, save_to)

def plot_consumer_cost_comparison(
    results_by_model: dict[str, dict],
    models: list[str] | tuple[str, ...] | None = None,
    title: str = "Consumer Cost Comparison",
    save_to: Path | str | None = None,
) -> plt.Figure:
    """Plot per-consumer costs for a user-selected set of models.

    Args:
        results_by_model: Mapping from model name to its solve result dictionary.
            Each result must contain a ``consumer_costs`` mapping.
        models: Model names to include. Defaults to all available models.
            Example: ``["R", "I"]`` or ``["C", "DP"]``.
        title: Figure title.
        save_to: Optional image path.

    Returns:
        Matplotlib figure containing grouped bar charts.
    """
    available = list(results_by_model)
    selected = list(available if models is None else models)

    missing = [model for model in selected if model not in results_by_model]
    if missing:
        raise KeyError(f"Unknown model(s): {missing}. Available models: {available}")

    consumer_ids = []
    for model in selected:
        for consumer_id in results_by_model[model]["consumer_costs"]:
            if consumer_id not in consumer_ids:
                consumer_ids.append(consumer_id)

    if not consumer_ids:
        raise ValueError("No consumer cost data found")

    fig, ax = plt.subplots(figsize=(max(12, len(consumer_ids) * 0.75), 6))
    x = np.arange(len(consumer_ids))
    width = 0.8 / len(selected)

    for offset, model in enumerate(selected):
        costs = results_by_model[model]["consumer_costs"]
        values = [costs.get(consumer_id, np.nan) for consumer_id in consumer_ids]
        ax.bar(
            x + (offset - (len(selected) - 1) / 2) * width,
            values,
            width,
            label=model,
        )

    ax.set_xlabel("Consumer")
    ax.set_ylabel("Cost (DKK)")
    ax.set_title(title)
    ax.set_xticks(x, consumer_ids, rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="Model")

    return _finish(fig, save_to)