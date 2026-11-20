"""Numerical checks for the dynamic-pricing model built in src/model.py."""


def pricing_diagnostics(owner, prices):
    """Check the payment identity and follower primal/dual optimality numerically."""
    primal_residual = 0.0
    complementarity = 0.0
    for entry in owner.member_kkt:
        row = entry["row"]
        residual = owner.m.getRow(row).getValue() - row.RHS
        violation = abs(residual) if row.Sense == "=" else (
            max(0.0, residual) if row.Sense == "<" else max(0.0, -residual)
        )
        primal_residual = max(primal_residual, violation)
        if entry["slack"] is not None:
            complementarity = max(complementarity, abs(entry["dual"].X * entry["slack"].X))
    stationarity = max((abs(row.Slack) for row in owner.con["member_stationarity"].values()), default=0.0)
    payment_residual = 0.0
    payments = {}
    for i, member in enumerate(owner.members):
        member_prices = prices[member.consumer_id] if isinstance(prices, dict) else prices
        direct = sum(member_prices[t] * (
            owner.var["load"][i, t].X + owner.var["p_ch"][i, t].X
            - owner.var["pv"][i, t].X - owner.var["p_dis"][i, t].X
        ) for t in owner.T)
        linear = owner.member_payment[i].getValue()
        payments[member.consumer_id] = linear
        payment_residual = max(payment_residual, abs(direct - linear))
    return {"member_payments": payments,
            "kkt_primal_residual": primal_residual,
            "kkt_stationarity_residual": stationarity,
            "kkt_complementarity_residual": complementarity,
            "payment_identity_residual": payment_residual}
