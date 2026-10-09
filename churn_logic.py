
def predict_churn(tenure, support_calls):
    if tenure < 6 and support_calls >= 4:
        return 1
    return 0
