import numpy as np
import matplotlib.pyplot as plt
import json

function_forms = [
    # fit points from a csv file or from a function into various polynomial forms
    "f(x) = A0 + A1*x + A2*x^2 + ... + An*x^n",
    "f(x) = A_{-n}*x^-n + ... + A_{-2}*x^-2 + A_{-1}/x + A0 + A1*x + A2*x^2 + ... + An*x^n",
    "f(x) = (A0 + A1*x + A2*x^2 + ... + An*x^n)/(B0 + B1*x + B2*x^2 + ... + Bn*x^n)"
]

# Settings -------------------------------------------
FORM_SETTING = 2 # = 0, 1 or 2
DEGREE = 1
NEGATIVE_DEGREE = - DEGREE

USE_CSV = True
CSV_FILE_LOCATION = "points.csv"
# INDEPENDENT_VARIABLE_COUNT = 1
# DEPENDENT_VARIABLE_COUNT = 1

RANGE_OVER_X = (-10, 10) # = (-10, 10)
NUM_OF_POINTS = 500 # = 500
def custom_function(x):
    return (2*x + 1)/3

NUMBER_OF_LOOPS = 5 * (DEGREE + FORM_SETTING) # = 5 * (DEGREE + FORM_SETTING)
EPOCHS_PER_LOOP = 2 ** (10 + DEGREE + FORM_SETTING) # = 2 ** (10 + DEGREE + FORM_SETTING)
STARTING_LEARNING_RATE = 0.0001 # = 0.0001
LEARNING_RATE_DAMPENING_RATE = 0.9 # = 0.9
LAMBDA_REGULARIZATION = 0.1 # = 0.1
ROUND_TO = 2 # = 2
#-----------------------------------------------------

if USE_CSV:
    data = np.loadtxt(CSV_FILE_LOCATION, delimiter=',')
    if data.ndim != 2 or data.shape[1] < 2: # for now
        raise ValueError("CSV file must have at least two columns (independent and dependent variables).")
    indep_list = data[:, 0]  # Independent variable (x)........................
    dep_list = data[:, -1]  # Dependent variable (y)...........................
else:
    indep_list = np.linspace(RANGE_OVER_X[0], RANGE_OVER_X[-1], NUM_OF_POINTS)
    dep_list = np.array([custom_function(x) for x in indep_list])

# Generate power terms based on FORM
if FORM_SETTING == 1:
    powers = np.arange(NEGATIVE_DEGREE, DEGREE + 1)
else:
    powers = np.arange(DEGREE + 1)
x_powers = np.power.outer(indep_list, powers)

# Create run_epoch function based on FORM
if FORM_SETTING == 2:
    def run_epoch(constants):
        A_constants, B_constants = constants
        diff = B_constants - desired_B

        B_sum = x_powers @ B_constants
        div = x_powers @ A_constants / B_sum

        error = div - dep_list
        common_term = 2 * error / B_sum
        An_deltas = -(common_term @ x_powers)
        Bn_deltas = (common_term * div) @ x_powers - 2 * LAMBDA_REGULARIZATION * diff

        return (
            np.sum(error ** 2) + LAMBDA_REGULARIZATION * np.sum(diff ** 2),
            np.vstack([An_deltas, Bn_deltas])
        )  # total_error, deltas

else:
    def run_epoch(constants):
        error = x_powers @ constants - dep_list
        return np.sum(error ** 2), -2 * (error @ x_powers)  # total_error, deltas

# START ----------------------------------------------------------------------------------------------------------------
record = {
    "function form": function_forms[FORM_SETTING],
    "error": float('inf')
}
for _ in range(NUMBER_OF_LOOPS):
    if FORM_SETTING == 2:
        constants = np.random.random((2, len(powers)))  # 2 rows: A_constants and B_constants
        desired_B = np.array([1] + [0] * (len(powers) - 1))
    else:
        constants = np.random.random(len(powers))  # 1 row: A_constants
    best_constants = constants.copy()
    learning_rate = STARTING_LEARNING_RATE

    # FIRST EPOCH ----------------------------------
    last_total_error, deltas = run_epoch(constants)
    constants += learning_rate * deltas

    # -----------------
    for _ in range(EPOCHS_PER_LOOP):
        total_error, deltas = run_epoch(constants)

        # Update constants if error decreases
        if total_error <= last_total_error:
            last_total_error = total_error
            best_constants = constants.copy()
            if total_error < 1e-12:
                break
            constants += learning_rate * deltas

        else:  # Last update was not good
            constants = best_constants.copy()
            learning_rate *= LEARNING_RATE_DAMPENING_RATE

    if last_total_error <= record["error"]:
        record.update({
            "error": last_total_error,
            "constants": best_constants,
        })

# Process constants: round to n decimal places and scale ---------------------------------------------------------------
constants = record["constants"]
rounded_constants = np.round(constants, ROUND_TO)

if FORM_SETTING == 2:
    smallest_non_zero = np.min(np.absolute(rounded_constants[np.nonzero(rounded_constants)]))
    rounded_constants = np.round(rounded_constants / smallest_non_zero, ROUND_TO)

record["constants"] = record["constants"].tolist()
record["constants-rounded"] = rounded_constants.tolist()

# Save constants to JSON
with open('constants.json', 'w') as f:
   json.dump(record, f, indent=4)

# DRAW -----------------------------------------------------------------------------------------------------------------
if FORM_SETTING == 2:
    def display_function(x):
        A_constants, B_constants = constants
        return np.sum(A_constants * x**powers) / np.sum(B_constants * x**powers)

    def estimated_display_function(x):
        A_rounded_constants, B_rounded_constants = rounded_constants
        return np.sum(A_rounded_constants * x ** powers) / np.sum(B_rounded_constants * x ** powers)
else:
    def display_function(x):
        return np.sum(constants * x ** powers)

    def estimated_display_function(x):
        return np.sum(rounded_constants * x ** powers)

x_values = np.linspace(min(indep_list), max(indep_list), NUM_OF_POINTS)
y_values1 = np.array([display_function(x) for x in x_values])
y_values2 = np.array([estimated_display_function(x) for x in x_values])

# Plot the function and the original data points
plt.figure(figsize=(8, 6))
plt.scatter(indep_list, dep_list, color='red', label='Original Data Points', alpha=0.7)
plt.plot(x_values, y_values1, color='blue', label='Fitted Function', linewidth=2)
plt.plot(x_values, y_values2, color='green', label='Rounded Fitted Function', linewidth=2)

# Add labels, legend, and grid
plt.xlabel('Independent Variable (x)')
plt.ylabel('Dependent Variable (f(x))')
plt.title('Fitted Rational Function')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()