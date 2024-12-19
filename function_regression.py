import numpy as np
import matplotlib.pyplot as plt
import json

function_forms = [
    # fit points from a csv file or from a function into various polynomial forms
    "f(x) = A0 + A1*x + A2*x^2 + ... + An*x^n",
    "f(x) = A_{-n}*x^-n + ... + A_{-2}*x^-2 + A_{-1}/x + A0 + A1*x + A2*x^2 + ... + An*x^n"
    "f(x) = (A0 + A1*x + A2*x^2 + ... + An*x^n)/(B0 + B1*x + B2*x^2 + ... + Bn*x^n)"
]

# Settings -------------------------------------------
FORM_SETTING = 2
DEGREE = 1
NEGATIVE_DEGREE = - DEGREE

USE_CSV = True
CSV_FILE_LOCATION = "points.csv"
# INDEPENDENT_VARIABLE_COUNT = 1
# DEPENDENT_VARIABLE_COUNT = 1

RANGE_OVER_X = (-10, 10)
NUM_OF_POINTS = 500
def custom_function(x):
    return (2*x + 1)/3

STARTING_LEARNING_RATE = 0.0001 # = 0.0001
LEARNING_RATE_DAMPENING_RATE = 0.9 # = 0.9
EPOCHS_PER_LOOP = 20000 # = 20000
NUMBER_OF_LOOPS = 10 # = 10
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
if FORM_SETTING == 0 or FORM_SETTING == 2:
    powers = np.arange(DEGREE + 1)  # 0 to DEGREE
    x_powers = np.power.outer(indep_list, powers)  # Shape: (len(indep_list), DEGREE + 1)
elif FORM_SETTING == 1:
    powers = np.concatenate((-np.arange(NEGATIVE_DEGREE, 0)[::-1], np.arange(DEGREE + 1)))
    x_powers = np.power.outer(indep_list, powers)  # Shape: (len(indep_list), NEGATIVE_DEGREE + DEGREE + 1)
else:
    raise ValueError("Unsupported FORM_SETTING value.")

def run_epoch(A_constants, B_constants=None):
    A_sum = x_powers @ A_constants
    if FORM_SETTING == 2:
        B_sum = x_powers @ B_constants
        current_output = A_sum / B_sum

        error = current_output - dep_list
        total_error = np.sum(error ** 2)
        #print("errors", error, total_error)

        common_term = 2 * error / B_sum
        An_deltas = -np.sum(common_term[:, None] * x_powers, axis=0)
        Bn_deltas = np.sum((common_term * A_sum / B_sum)[:, None] * x_powers, axis=0)
        # print("deltas", An_deltas, Bn_deltas)

        return total_error, An_deltas, Bn_deltas

    current_output = A_sum
    #print("sums", A_sum, current_output)

    error = current_output - dep_list
    total_error = np.sum(error ** 2)
    #print("errors", error, total_error)

    common_term = 2 * error
    An_deltas = -np.sum(common_term[:, None] * x_powers, axis=0)
    # print("deltas", An_deltas)

    return total_error, An_deltas, None


record = {
    "function form": function_forms[FORM_SETTING],
    "error": float('inf')
}
for i in range(NUMBER_OF_LOOPS):
    # print(f"loop {i+1}:")
    A_constants = np.random.random(len(powers))
    if FORM_SETTING == 2:
        B_constants = np.random.random(len(powers))
    else:
        B_constants = None
    best_A_constants = A_constants.copy()
    best_B_constants = B_constants.copy() if B_constants is not None else None
    # print("constants", A_constants, B_constants)
    learning_rate = STARTING_LEARNING_RATE

    # FIRST EPOCH ----------------------------------
    results = run_epoch(A_constants, B_constants)
    last_total_error = results[0]
    A_constants += learning_rate * results[1]
    if FORM_SETTING == 2:
        B_constants += learning_rate * results[2]
    # print("new constants", A_constants, B_constants)

    # -----------------

    for epoch in range(EPOCHS_PER_LOOP):
        results = run_epoch(A_constants, B_constants)

        # Update constants if error decreases
        if results[0] <= last_total_error:
            last_total_error = results[0]
            best_A_constants = A_constants.copy()
            A_constants += learning_rate * results[1]
            if FORM_SETTING == 2:
                best_B_constants = B_constants.copy()
                B_constants += learning_rate * results[2]
            # print("new constants", A_constants, B_constants)
        else:  # Last update was not good
            learning_rate *= LEARNING_RATE_DAMPENING_RATE
            if learning_rate < 1e-12:
                break

            A_constants = best_A_constants.copy()
            if FORM_SETTING == 2:
                B_constants = best_B_constants.copy()
            # print("corrected constants", A_constants, B_constants)

        # print(f"Epoch {epoch}, Total Error: {results[0]}")

    if last_total_error <= record["error"]:
        record = {
            "An": best_A_constants,
            "Bn": best_B_constants,
            "error": last_total_error
        }

# Process constants: round to n decimal places and scale ---------------------------------------------------------------
best_A_constants = record["An"]
best_B_constants = record["Bn"]

rounded_A_constants = np.round(best_A_constants, ROUND_TO)


if FORM_SETTING == 2:
    rounded_B_constants = np.round(best_B_constants, ROUND_TO)

    # Combine both arrays to find the smallest non-zero value
    all_constants = np.concatenate([rounded_A_constants, rounded_B_constants])
    smallest_non_zero = np.min(all_constants[np.nonzero(all_constants)])  # Find smallest non-zero value
    # print(f"{smallest_non_zero=}")

    # Scale constants so that the smallest non-zero value becomes 1
    scale_factor = np.absolute(1 / smallest_non_zero) if smallest_non_zero != 0 else 1
    scaled_A_constants = np.round(rounded_A_constants * scale_factor, ROUND_TO)
    scaled_B_constants = np.round(rounded_B_constants * scale_factor, ROUND_TO)

    record["An"] = record["An"].tolist()
    record["Bn"] = record["Bn"].tolist()
    record["An-rounded"] = scaled_A_constants.tolist()
    record["Bn-rounded"] = scaled_B_constants.tolist()
else:
    record["An"] = record["An"].tolist()
    record["An-rounded"] = rounded_A_constants.tolist()

# Save constants to JSON
with open('constants.json', 'w') as f:
   json.dump(record, f, indent=4)

# DRAW -----------------------------------------------------------------------------------------------------------------
def polynomial_function(x):
    return np.sum(best_A_constants * x**powers)

def estimated_polynomial_function(x):
    return np.sum(rounded_A_constants * x**powers)

def rational_function(x):
    return np.sum(best_A_constants * x**powers) / np.sum(best_B_constants * x**powers)

def estimated_rational_function(x):
    return np.sum(scaled_A_constants * x**powers) / np.sum(scaled_B_constants * x**powers)

x_values = np.linspace(min(indep_list), max(indep_list), 500)
if FORM_SETTING == 2:
    y_values1 = np.array([rational_function(x) for x in x_values])
    y_values2 = np.array([estimated_rational_function(x) for x in x_values])
else:
    y_values1 = np.array([polynomial_function(x) for x in x_values])
    y_values2 = np.array([estimated_polynomial_function(x) for x in x_values])

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