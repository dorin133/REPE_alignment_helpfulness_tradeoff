import matplotlib.pyplot as plt
import numpy as np

# mean:
international_law_mean = {...}

# std:
international_law_std = {...}

# mean:
high_school_computer_science_mean = {...}

# std:
high_school_computer_science_std = {...}

#mean:
medical_genetics_mean = {...}

# std:
medical_genetics_std = {...}


def plot_dataset_results(dataset_name, mean, std, color='blue'):
    print('dataset_name:', dataset_name)
    print('mean:', mean)
    print('std:', std)
    print('________')
    x = list(mean.keys())
    y = list(mean.values())
    yerr = list(std.values())

    plt.errorbar(x, y, yerr=yerr, label=dataset_name, color=color)

    # Fit a line to the data
    index_of_zero = x.index(0)
    x_p, y_p = x[index_of_zero:], y[index_of_zero:]
    index_of_5 = x_p.index(5)
    coefficients_p = np.polyfit(x_p[:index_of_5+1], y_p[:index_of_5+1], 1)
    fit_function_p = np.poly1d(coefficients_p)
    x_n, y_n = x[:index_of_zero+1], y[:index_of_zero+1]
    index_of_minus_5 = x_n.index(-5)
    coefficients_n = np.polyfit(x_n[index_of_minus_5:], y_n[index_of_minus_5:], 1)
    fit_function_n = np.poly1d(coefficients_n)

    # Plot the fitted line
    plt.plot(x_p, fit_function_p(x_p), '--', label=f'{dataset_name} fit', color='black')
    plt.plot(x_n, fit_function_n(x_n), '--', label=f'{dataset_name} fit', color='black')

    equation_p = f'y_pos = {fit_function_p[1]:.2f}x + {fit_function_p[0]:.2f}'
    equation_n = f'y_neg = {fit_function_n[1]:.2f}x + {fit_function_n[0]:.2f}'

    equation = equation_p + '\n' + equation_n

    plt.annotate(equation, xy=(0.75, 0.15), xycoords='axes fraction', fontsize=12, color=color)

    plt.title(f'final layer representation change norm of llama-2-13b. Dataset: {dataset_name}')
    plt.xlabel(r'$r_e$')
    plt.ylabel(r'$|U\cdot \delta_{r_e}(q)|$')
    plt.show()


plot_dataset_results('international_law', international_law_mean, international_law_std, color='b')
plot_dataset_results('high_school_computer_science', high_school_computer_science_mean, high_school_computer_science_std, color='r')
plot_dataset_results('medical_genetics', medical_genetics_mean, medical_genetics_std, color='y')
