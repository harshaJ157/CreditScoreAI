def generate_recommendations(profile: dict) -> list[str]:
    """Rule-based educational guidance; not a score calculation."""
    tips = []

    utilization = profile.get("Credit_Utilization_Ratio")
    if isinstance(utilization, (int, float)) and utilization > 30:
        tips.append(
            "Your entered credit-utilization ratio is relatively high. "
            "If possible, consider reducing revolving balances and avoid using "
            "a large share of available credit."
        )

    payment_min = str(profile.get("Payment_of_Min_Amount", "")).strip().lower()
    if payment_min in {"no", "not paid", "minimum not paid"}:
        tips.append(
            "Review whether minimum payments are being made on time. "
            "Paying bills by their due dates can help avoid additional fees "
            "and negative repayment records."
        )

    behaviour = str(profile.get("Payment_Behaviour", "")).lower()
    if "spent" in behaviour and ("high" in behaviour or "large" in behaviour):
        tips.append(
            "Review spending and repayment patterns against your budget. "
            "This field is dataset-specific and is not by itself proof of poor credit."
        )

    debt = profile.get("Outstanding_Debt")
    if isinstance(debt, (int, float)) and debt > 0:
        tips.append(
            "Review your outstanding balances and repayment commitments. "
            "Choose a repayment plan that fits your budget."
        )

    if not tips:
        tips.append(
            "Keep reviewing your credit report for errors, pay bills on time, "
            "and borrow only what you can comfortably repay."
        )

    tips.append(
        "These are general educational suggestions, not a guarantee of a score increase. "
        "Check your official credit report for bureau-specific information."
    )
    return tips
