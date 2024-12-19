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

# Create run_epoch function based on FORM
if FORM_SETTING == 2:
    def run_epoch(constants):
        A_constants = constants[0, :]  # Row 0: A_constants
        B_constants = constants[1, :]  # Row 1: B_constants
        A_sum = x_powers @ A_constants
        B_sum = x_powers @ B_constants
        current_output = A_sum / B_sum
        #print("sums", A_sum, B_sum, current_output)

        error = current_output - dep_list
        total_error = np.sum(error ** 2)
        #print("errors", error, total_error)

        common_term = 2 * error / B_sum
        An_deltas = -np.sum(common_term[:, None] * x_powers, axis=0)
        Bn_deltas = np.sum((common_term * A_sum / B_sum)[:, None] * x_powers, axis=0)
        deltas = np.vstack([An_deltas, Bn_deltas])  # Stack deltas into a 2D array
        #print("deltas", deltas)

        return total_error, deltas
else:
    def run_epoch(constants):
        current_output = x_powers @ constants
        #print("sums", current_output)

        error = current_output - dep_list
        total_error = np.sum(error ** 2)
        #print("errors", error, total_error)

        common_term = 2 * error
        deltas = -np.sum(common_term[:, None] * x_powers, axis=0)
        #print("deltas", deltas)

        return total_error, deltas

# START ----------------------------------------------------------------------------------------------------------------
record = {
    "function form": function_forms[FORM_SETTING],
    "error": float('inf')
}
for i in range(NUMBER_OF_LOOPS):
    #print(f"loop {i+1}:")
    if FORM_SETTING == 2:
        constants = np.random.random((2, len(powers)))  # 2 rows: A_constants and B_constants
    else:
        constants = np.random.random(len(powers))  # 1 row: A_constants
    best_constants = constants.copy()
    #print("constants", constants)
    learning_rate = STARTING_LEARNING_RATE

    # FIRST EPOCH ----------------------------------
    last_total_error, deltas = run_epoch(constants)
    constants += learning_rate * deltas
    #print("new constants", constants)

    # -----------------
    for epoch in range(EPOCHS_PER_LOOP):
        total_error, deltas = run_epoch(constants)

        # Update constants if error decreases
        if total_error <= last_total_error:
            last_total_error = total_error
            best_constants = constants.copy()
            if total_error < 1e-12:
                break
            constants += learning_rate * deltas
            #print("new constants", constants)

        else:  # Last update was not good
            constants = best_constants.copy()
            learning_rate *= LEARNING_RATE_DAMPENING_RATE
            if learning_rate < 1e-12:
                break
            #print("corrected constants", constants)

        #print(f"Epoch {epoch}, Total Error: {total_error}")

    if last_total_error <= record["error"]:
        record.update({
            "error": last_total_error,
            "constants": best_constants,
        })

# Process constants: round to n decimal places and scale ---------------------------------------------------------------
constants = record["constants"]
rounded_constants = np.round(constants, ROUND_TO)

if FORM_SETTING == 2:
    smallest_non_zero = np.min(rounded_constants[np.nonzero(rounded_constants)])
    # Scale constants so that the smallest non-zero value becomes 1
    scale_factor = np.absolute(1 / smallest_non_zero)
    rounded_constants = np.round(rounded_constants * scale_factor, ROUND_TO)

record["constants"] = record["constants"].tolist()
record["constants-rounded"] = rounded_constants.tolist()

# Save constants to JSON
with open('constants.json', 'w') as f:
   json.dump(record, f, indent=4)

# DRAW -----------------------------------------------------------------------------------------------------------------
if FORM_SETTING == 2:
    def display_function(x):
        return np.sum(constants[0, :] * x**powers) / np.sum(constants[1, :] * x**powers)

    def estimated_display_function(x):
        return np.sum(rounded_constants[0, :] * x ** powers) / np.sum(rounded_constants[1, :] * x ** powers)
else:
    def display_function(x):
        return np.sum(constants * x ** powers)

    def estimated_display_function(x):
        return np.sum(rounded_constants * x ** powers)

x_values = np.linspace(min(indep_list), max(indep_list), 500)
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