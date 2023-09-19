# import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2


def ConfidenceInterval(alpha, F, circuitlength, time):
    Chi_lower_bound = chi2.ppf(alpha / 2, 2 * F)
    Chi_upper_bound = chi2.ppf(1 - (alpha / 2), 2 * F + 2)

    T = circuitlength * time

    lower_bound = Chi_lower_bound / (2 * T)
    upper_bound = Chi_upper_bound / (2 * T)

    return lower_bound, upper_bound


if __name__ == "__main__":
    circuitlength_EHV = 2310  # unit: km ## Q2: 2310, Q3: 2471
    circuitlength_HV = 3329  # unit: km ## Q2: 3329, Q3: 4078
    failures_EHV = 25  # Q2: 25, Q3: 42
    failures_HV = 51  # Q2: 51, Q3: 167
    time = 5  # unit: years

    alpha = 0.05

    ## Calculate the bounds of the confidence interval of EHV statistics
    lower_bound_EHV, upper_bound_EHV = ConfidenceInterval(
        alpha, failures_EHV, circuitlength_EHV, time
    )
    ## Calculate the bounds of the confidence interval of HV statistics
    lower_bound_HV, upper_bound_HV = ConfidenceInterval(
        alpha, failures_HV, circuitlength_HV, time
    )
    ## Calculate the bounds of the confidence interval of HV statistics
    lower_bound_OHL, upper_bound_OHL = ConfidenceInterval(
        alpha, failures_EHV + failures_HV, circuitlength_EHV + circuitlength_HV, time
    )

    print("EHV confidence interval:", lower_bound_EHV, upper_bound_EHV)
    print("HV confidence interval:", lower_bound_HV, upper_bound_HV)
    print("OHL confidence interval:", lower_bound_OHL, upper_bound_OHL)
    # boxplot()

    f_EHV = failures_EHV / (circuitlength_EHV * time)
    f_HV = failures_HV / (circuitlength_HV * time)
    f_OHL = (failures_EHV + failures_HV) / ((circuitlength_EHV + circuitlength_HV) * time)

    print("f_EHV:", f_EHV, "[cctkm * yr]")
    print("f_HV:", f_HV, "[cctkm * yr]")
    print("f_OHL:", f_OHL, "[cctkm * yr]")
