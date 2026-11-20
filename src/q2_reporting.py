"""Q2 report figures, metric definitions and tables from the code's saved CSVs."""
from pathlib import Path
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


NUMERIC = ["cost_DKK", "cost_above_C_DKK", "manager_balance_DKK", "peak_import_kW",
           "cap_exceedance_kWh", "max_cap_exceedance_kW", "hours_above_cap",
           "objective_bound_DKK", "mip_gap", "runtime_seconds", "members_worse_off_vs_R",
           "members_worse_off_vs_I", "members_worse_off_vs_best"]
COLORS = {"R":"#858b94", "I":"#cc873a", "C":"#202b3b", "DP_cap":"#3279ad",
          "DP_bb":"#39856a", "DP_ir_individual":"#bd5575"}


def _name(label):
    return {"DP_cap":"DP: cap", "DP_bb":"DP: +BB",
            "DP_ir_individual":"DP: +BB+IR\nindividual prices"}.get(label, label)


def _status(row):
    if pd.notna(row["cost_DKK"]):
        return "Optimal" if row["status"] == "Optimal" else "Incumbent"
    text = str(row["status"]).lower()
    return "Time limit; no incumbent" if "time limit" in text else str(row["status"])


def _save(fig, folder, name):
    fig.savefig(folder/name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _input_signature(folder):
    return hashlib.sha256(b"".join((folder/name).read_bytes() for name in
        ["summary.csv", "member_costs.csv", "schedule.csv", "members_schedule.csv"])).hexdigest()


def _load(folder):
    frames = {name:pd.read_csv(folder/f"{name}.csv", keep_default_na=False)
              for name in ["summary", "member_costs", "schedule", "members_schedule"]}
    if "price_scope" not in frames["summary"]:
        frames["summary"]["price_scope"] = "uniform"
    # Legacy uniform BB+IR runs are retired from the report comparison.
    legacy_uniform = ((frames["summary"]["model"] == "DP_ir")
                      & (frames["summary"]["price_scope"] == "uniform"))
    retired = set(frames["summary"].loc[legacy_uniform,"model"])
    individual_ir = ((frames["summary"]["model"] == "DP_ir")
                     & (frames["summary"]["price_scope"] == "individual")).any()
    for frame in frames.values():
        frame.drop(frame.index[frame["model"].isin(retired)],inplace=True)
        if individual_ir:
            frame["model"] = frame["model"].replace({"DP_ir":"DP_ir_individual"})
    for col in NUMERIC:
        frames["summary"][col] = pd.to_numeric(frames["summary"][col], errors="coerce")
    for key in ["member_costs", "schedule", "members_schedule"]:
        for col in frames[key].columns:
            if col not in ["model", "consumer_id", "archetype"]:
                frames[key][col] = pd.to_numeric(frames[key][col], errors="coerce")
    return frames


def diagnose_fixed_dispatch(folder, community="base"):
    """Minimize IR shortfalls by repricing a saved BB (or C) dispatch.

    Dispatch is fixed ONLY in this diagnostic, not in the DP experiments. Positive
    slacks describe one optimal shortfall allocation, not global blocking members.
    """
    from gurobipy import GRB, quicksum
    from src.data_loader import load_community
    from src.model import EnergyCommunityModel
    f = _load(folder)
    meta = f["summary"].iloc[-1]
    available = set(f["members_schedule"]["model"])
    source = "DP_bb" if "DP_bb" in available else "C"
    if source == "DP_bb":
        meta = f["summary"].query("model == @source").iloc[0]
    data = load_community(community, day=str(meta["day"]))
    scope = str(meta.get("price_scope", "uniform"))
    config = {"budget_balance":True, "individual_rationality":False,
              "price_scope":scope, "daily_energy_rule":str(meta["daily_energy_rule"]),
              "discount":float(meta["discount"]), "penalty":float(meta["penalty_DKK_per_kWh"]),
              "cap_factor":float(meta["cap_factor"]), "time_limit":30,
              "price_bounds":tuple(None if meta[k] == "" else float(meta[k]) for k in ["price_min","price_max"])}
    model = EnergyCommunityModel(data, "DP", config)
    model.build_model()
    physical = f["members_schedule"].query("model == @source").set_index(["consumer_id","hour"])
    grid = f["schedule"].query("model == @source").set_index("hour")
    if set(physical.index.get_level_values(0)) != set(data.member_ids):
        raise ValueError("Diagnostic community does not match saved member IDs")
    for i, member in enumerate(data.members):
        for t in model.T:
            for key, column in [("load","load"),("pv","pv"),("dev","dev"),
                                ("p_ch","p_ch"),("p_dis","p_dis"),("soc","soc")]:
                value = float(physical.loc[(member.consumer_id,t),column])
                model.var[key][i,t].LB = value
                model.var[key][i,t].UB = value
    for t in model.T:
        imp, exp = float(grid.loc[t,"import_kW"]), float(grid.loc[t,"export_kW"])
        for key, value in {"P_imp":imp, "P_exp":exp, "P_below":min(imp,model.cap_limit[t]),
                           "P_exc":max(0,imp-model.cap_limit[t])}.items():
            model.var[key][t].LB = value
            model.var[key][t].UB = value
    costs = f["member_costs"].set_index(["model","consumer_id"])
    reference = str(meta["IR_reference"])
    benchmark = {}
    for member in data.members:
        cid = member.consumer_id
        options = [float(costs.loc[(mode,cid),"cost_DKK"])
                   for mode in (["R","I"] if reference == "both" else [reference])]
        benchmark[cid] = min(options)
    shortfall = model.m.addVars(len(data.members), lb=0, name="diagnostic_IR_shortfall")
    for i, member in enumerate(data.members):
        model.m.addConstr(model.member_total_cost[i] <= benchmark[member.consumer_id]+shortfall[i])
    model.m.setObjective(quicksum(shortfall.values()),GRB.MINIMIZE)
    model.m.Params.DualReductions = 0
    model.m.optimize()
    has_solution = model.m.SolCount > 0
    status = "Optimal" if model.m.Status == GRB.OPTIMAL else f"Solver status {model.m.Status}"
    report = folder/"report"
    report.mkdir(exist_ok=True)
    row = {"scope":"fixed-dispatch repricing only", "source_model":source,
           "price_scope":scope, "IR_reference":reference, "status":status,
           "total_shortfall_DKK":model.m.ObjVal if has_solution else None,
           "objective_bound_DKK":model.m.ObjBound if has_solution else None,
           "input_signature":_input_signature(folder),
           "interpretation":"One minimum-shortfall allocation at fixed dispatch; not a global blocker proof."}
    pd.DataFrame([row]).to_csv(report/"diagnostic_summary.csv",index=False)
    rows = []
    if has_solution:
        for i, member in enumerate(data.members):
            rows.append({"consumer_id":member.consumer_id, "archetype":member.archetype,
                         "outside_cost_DKK":benchmark[member.consumer_id],
                         "repriced_cost_DKK":model.member_total_cost[i].getValue(),
                         "shortfall_DKK":shortfall[i].X,
                         "scope":"fixed-dispatch repricing only"})
    pd.DataFrame(rows,columns=["consumer_id","archetype","outside_cost_DKK",
                              "repriced_cost_DKK","shortfall_DKK","scope"]).to_csv(
                                  report/"diagnostic_member_shortfalls.csv",index=False)
    model.m.dispose()


def generate_q2_report(output, comparison_dirs=(), diagnose_ir=False, community="base"):
    """Produce report artifacts without re-solving the mandatory DP experiments."""
    output = Path(output)
    if not (output/"summary.csv").exists():
        raise FileNotFoundError(f"No Q2 summary.csv in {output}; solve the experiment first")
    f = _load(output)
    base_meta = f["summary"].iloc[-1]
    for directory in comparison_dirs:
        other = _load(Path(directory))
        meta = other["summary"].iloc[-1]
        for col in ["day","daily_energy_rule","discount","penalty_DKK_per_kWh","cap_factor","IR_reference"]:
            if str(meta[col]) != str(base_meta[col]):
                raise ValueError(f"Comparison differs in {col}: {directory}")
        # Validate reference costs as well; do not silently merge different data.
        for mode in ["R","I","C"]:
            a = float(f["summary"].query("model == @mode").iloc[0]["cost_DKK"])
            b = float(other["summary"].query("model == @mode").iloc[0]["cost_DKK"])
            if abs(a-b) > 1e-5:
                raise ValueError(f"Comparison has different {mode} benchmark: {directory}")
        labels = other["summary"].loc[other["summary"]["model"].str.startswith("DP_"),"model"]
        scope = str(meta.get("price_scope","uniform"))
        mapping = {label:(label if label.endswith("_"+scope) else f"{label}_{scope}") for label in labels}
        if set(mapping.values()) & set(f["summary"]["model"]):
            raise ValueError("Duplicate comparison model labels")
        for key in f:
            extra = other[key][other[key]["model"].isin(mapping)].copy()
            extra["model"] = extra["model"].map(mapping)
            f[key] = pd.concat([f[key],extra],ignore_index=True)
    report = output/"report"
    report.mkdir(exist_ok=True)
    s = f["summary"].copy()
    r_cost = float(s.query("model == 'R'").iloc[0]["cost_DKK"])
    c_cost = float(s.query("model == 'C'").iloc[0]["cost_DKK"])
    i_cost = float(s.query("model == 'I'").iloc[0]["cost_DKK"])
    r_exc = float(s.query("model == 'R'").iloc[0]["cap_exceedance_kWh"])
    c_exc = float(s.query("model == 'C'").iloc[0]["cap_exceedance_kWh"])
    reference = str(base_meta["IR_reference"])
    gain_column = {"R":"gain_vs_R_DKK","I":"gain_vs_I_DKK","both":"gain_vs_best_DKK"}[reference]
    s["saving_vs_R_DKK"] = r_cost-s["cost_DKK"]
    s["saving_vs_I_DKK"] = i_cost-s["cost_DKK"]
    s["C_value_recovered_pct"] = 100*(r_cost-s["cost_DKK"])/(r_cost-c_cost) if r_cost-c_cost>1e-8 else np.nan
    s["C_exceedance_reduction_recovered_pct"] = 100*(r_exc-s["cap_exceedance_kWh"])/(r_exc-c_exc) if r_exc-c_exc>1e-8 else np.nan
    s["display_status"] = s.apply(_status,axis=1)
    s["members_worse_off_vs_selected_reference"] = s["members_worse_off_vs_"+({"both":"best"}.get(reference,reference))]
    s.to_csv(report/"comparison_metrics.csv",index=False)
    impact = f["member_costs"].copy()
    impact["selected_reference"] = reference
    impact["gain_vs_selected_reference_DKK"] = impact[gain_column]
    impact["observed_shortfall_DKK"] = (-impact[gain_column]).clip(lower=0)
    impact.loc[impact["observed_shortfall_DKK"] < 1e-6,"observed_shortfall_DKK"] = 0.0
    impact["interpretation"] = "Loss in this saved settlement; not a proven blocking member"
    impact.to_csv(report/"member_impact.csv",index=False)
    costs = s.set_index("model")["cost_DKK"]
    increments = []
    for step,previous in [("DP_cap","C"),("DP_bb","DP_cap"),("DP_ir_individual","DP_bb")]:
        if step not in costs.index:continue
        known = previous in costs.index and pd.notna(costs[step]) and pd.notna(costs[previous])
        increments.append({"step":step,"reference":previous,
            "observed_increment_DKK":costs[step]-costs[previous] if known else None,
            "status":s.set_index("model").loc[step,"display_status"],
            "price_scope_changed": bool(previous in costs.index and
                s.set_index("model").loc[step,"price_scope"] != s.set_index("model").loc[previous,"price_scope"]),
            "interpretation":(
                "Joint change: add IR and use individual prices; not an isolated cost of IR"
                if step == "DP_ir_individual" and previous in costs.index and
                   s.set_index("model").loc[step,"price_scope"] != s.set_index("model").loc[previous,"price_scope"]
                else "Observed incumbent difference; check solver status/gap"
            ) if known else "Unresolved; no cost or infeasibility claim"})
    pd.DataFrame(increments).to_csv(report/"requirement_costs.csv",index=False)
    definitions = [
        ("Community cost","DKK/day","External grid bill plus all member operating/discomfort costs; excludes internal transfers","C is the efficiency benchmark"),
        ("Saving vs R / I","DKK/day","C_R-C_M / C_I-C_M","Positive means the community gains"),
        ("C value recovered","%","100*(C_R-C_M)/(C_R-C_C)","Undefined unless C_R>C_C; 100% recovers C's saving"),
        ("Cap exceedance","kWh/day","sum_t max(0,import_t-cap_t)*1 hour","R/I: sum of member imports; C/DP: connection-point import"),
        ("Peak import","kW","max_t import_t","Compare with R and C"),
        ("Member gain","DKK/day","outside_cost_i - member_cost_i","Outside cost is R, I, or min(R,I), as selected"),
        ("Observed shortfall","DKK/day","max(0,member_cost_i-outside_cost_i)","Compensation needed at this settlement, not a global minimum"),
        ("Manager balance","DKK/day","member payments collected - external grid bill","Zero is exact budget balance"),
        ("Added requirement cost","DKK/day","observed cost after requirement - observed cost before","Only compare available solutions; check solver gaps"),
        ("Fixed-dispatch minimum shortfall","DKK/day","min sum_i v_i, cost_i<=outside_cost_i+v_i, with exact BB and fixed device schedules","Diagnostic only: prices reoptimized; dispatch fixed; not a global blocking-member proof"),
    ]
    pd.DataFrame(definitions,columns=["metric","unit","definition","reference_and_sign"]).to_csv(report/"metric_definitions.csv",index=False)

    # Small table ready to include in a report. Unknown outcomes stay unknown.
    headers=["Model","Status","Cost\nDKK","C value\nrecovered %","Peak\nkW","Excess\nkWh","Manager\nbalance DKK","Members\nworse off"]
    table=[]
    for _,row in s.iterrows():
        fmt=lambda key: "—" if pd.isna(row[key]) else f"{(0.0 if abs(row[key]) < .005 else row[key]):.2f}"
        status=row["display_status"].replace("; ","\n")
        table.append([_name(row["model"]),status,fmt("cost_DKK"),fmt("C_value_recovered_pct"),
                      fmt("peak_import_kW"),fmt("cap_exceedance_kWh"),fmt("manager_balance_DKK"),
                      "—" if pd.isna(row["members_worse_off_vs_selected_reference"]) else str(int(row["members_worse_off_vs_selected_reference"]))])
    fig,ax=plt.subplots(figsize=(15,1.8+0.48*len(table)));ax.axis("off")
    tab=ax.table(cellText=table,colLabels=headers,cellLoc="center",loc="center",
                 colWidths=[.15,.23,.09,.11,.08,.08,.12,.12])
    tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,2.0)
    for (i,j),cell in tab.get_celld().items():
        cell.set_edgecolor("#d5dce3")
        if i==0:cell.set_facecolor("#253c50");cell.set_text_props(color="white",weight="bold")
        elif i%2:cell.set_facecolor("#f0f4f7")
    ax.set_title(f"Q2 comparison — day {base_meta['day']}; participation reference: {reference}",pad=18,weight="bold")
    fig.text(.03,.02,"Blank entries mean no incumbent, not zero cost. Member allocations depend on the selected optimal prices.",fontsize=9)
    _save(fig,report,"comparison_table.png")

    fig,axes=plt.subplots(2,2,figsize=(14,8),layout="constrained")
    for ax,col,title,ylabel in [(axes[0,0],"cost_DKK","Community cost","DKK/day"),
                               (axes[0,1],"cap_exceedance_kWh","Congestion remaining","kWh/day"),
                               (axes[1,0],"manager_balance_DKK","Manager surplus or deficit","DKK/day"),
                               (axes[1,1],"members_worse_off_vs_selected_reference","Members worse off","Number of members")]:
        x=np.arange(len(s));vals=s[col].to_numpy(dtype=float)
        good=np.isfinite(vals)
        ax.bar(x[good],vals[good],color=[COLORS.get(v,"#85539a") for v in s.loc[good,"model"]])
        ax.set_xticks(x,[_name(v) for v in s["model"]],rotation=25,ha="right",fontsize=8)
        ax.set_title(title);ax.set_ylabel(ylabel);ax.axhline(0,color="#52606d",lw=.7)
        ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
        for j in x[~good]:ax.text(j,.03,"No incumbent",rotation=90,ha="center",va="bottom",fontsize=8,transform=ax.get_xaxis_transform())
        if col=="cost_DKK":ax.axhline(c_cost,color="black",ls="--",lw=1,label="C benchmark");ax.legend(fontsize=8)
    fig.suptitle("What changes when budget balance and participation are required?",weight="bold")
    _save(fig,report,"community_comparison.png")

    # Per-member gains against both outside options, preserving archetype order.
    labels=[v for v in s["model"] if v=="C" or v.startswith("DP_")]
    ids=list(dict.fromkeys(impact["consumer_id"]))
    matrices=[]
    for col in ["gain_vs_R_DKK","gain_vs_I_DKK"]:
        matrices.append(impact.pivot(index="consumer_id",columns="model",values=col).reindex(index=ids,columns=labels).to_numpy(dtype=float))
    finite=np.concatenate([a[np.isfinite(a)] for a in matrices]);limit=max(1,float(abs(finite).max()))
    fig,axes=plt.subplots(1,2,figsize=(max(13,len(labels)*2.3),max(6,len(ids)*.43)),layout="constrained",sharey=True)
    cmap=plt.get_cmap("RdBu").copy();cmap.set_bad("#e5e7eb")
    for ax,values,title in zip(axes,matrices,["Gain compared with R","Gain compared with I"]):
        im=ax.imshow(np.ma.masked_invalid(values),cmap=cmap,norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),aspect="auto")
        ax.set_xticks(range(len(labels)),[_name(v) for v in labels],rotation=25,ha="right")
        ax.set_yticks(range(len(ids)),ids);ax.set_title(title)
        for i in range(len(ids)):
            for j in range(len(labels)):
                v=values[i,j];ax.text(j,i,"—" if not np.isfinite(v) else f"{(0.0 if abs(v) < .05 else v):.1f}",ha="center",va="center",fontsize=8,color="white" if np.isfinite(v) and abs(v)>.6*limit else "#17212b")
    fig.colorbar(im,ax=axes,label="Gain (DKK/day): blue = benefit; red = loss",shrink=.85)
    fig.suptitle("Who benefits under each pricing step?",weight="bold")
    _save(fig,report,"member_gains.png")

    fig,axes=plt.subplots(1,2,figsize=(14,4.6),layout="constrained",sharey=True)
    for ax,chosen,title in [(axes[0],["R","I","C"],"Q1 reference models"),
                            (axes[1],["C"]+[v for v in labels if v.startswith("DP_")],"Q2 pricing variants")]:
        for label in chosen:
            rows=f["schedule"].query("model == @label").sort_values("hour")
            if rows.empty:continue
            ax.step(rows["hour"],rows["import_kW"],where="mid",label=_name(label).replace("\n"," "),
                    color=COLORS.get(label,"#85539a"),ls="--" if label=="C" else "-",lw=1.5)
        cap=f["schedule"].query("model == 'C'").sort_values("hour")
        ax.step(cap["hour"],cap["cap_kW"],where="mid",color="#b13232",ls=":",label="DSO cap",lw=2)
        ax.set_xlabel("Hour");ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel("Metered import (kW)")
    fig.suptitle("Flexibility delivered to the DSO; unavailable schedules are omitted",weight="bold")
    _save(fig,report,"import_profiles.png")

    prices=f["members_schedule"].copy()
    present=set(prices.loc[prices["internal_price_DKK_per_kWh"].notna(),"model"])
    chosen=next((v for v in reversed(labels) if v.startswith("DP_") and v in present),None)
    if chosen:
        values=prices.query("model == @chosen").pivot(index="consumer_id",columns="hour",values="internal_price_DKK_per_kWh").reindex(ids)
        fig,ax=plt.subplots(figsize=(12,max(5,len(ids)*.34)),layout="constrained")
        im=ax.imshow(values.to_numpy(),aspect="auto",cmap="viridis")
        ax.set_yticks(range(len(ids)),ids);ax.set_xticks(range(0,24,2),range(0,24,2));ax.set_xlabel("Hour")
        ax.set_title(f"Posted prices: {_name(chosen).replace(chr(10),' ')} — one selected optimal settlement")
        fig.colorbar(im,ax=ax,label="DKK/kWh")
        _save(fig,report,"posted_prices.png")

    if diagnose_ir:
        diagnose_fixed_dispatch(output,community)
        d=pd.read_csv(report/"diagnostic_member_shortfalls.csv")
        if not d.empty:
            fig,ax=plt.subplots(figsize=(12,4.6),layout="constrained")
            ax.bar(d["consumer_id"],d["shortfall_DKK"],color="#bb5653")
            ax.tick_params(axis="x",rotation=45);ax.set_ylabel("IR shortfall (DKK/day)")
            status=pd.read_csv(report/"diagnostic_summary.csv").iloc[0]["status"]
            qualifier="Minimum" if status == "Optimal" else "Best found"
            ax.set_title(f"{qualifier} total shortfall from repricing the fixed saved dispatch\nOne allocation; not a global blocking-member proof")
            ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
            _save(fig,report,"fixed_dispatch_shortfalls.png")
    print(f"Q2 report figures and tables saved to: {report}")
    return report
