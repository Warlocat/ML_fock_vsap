import matplotlib.pyplot as plt
import numpy as np

def plot_performance(pred, true, name = 'y', title = '', lim = None):
    

    x = pred.flatten()
    y = true.flatten()
    plt.subplots(figsize=(4, 4))
    plt.scatter(x, y, label = 'ML vs. True', s = 2)
    plt.plot([min(x), max(x)], [min(x), max(x)], color = 'red', linestyle = '--',label = 'Reference')
    plt.xlabel(f'{name} ML')
    plt.ylabel(f'{name} True')
    plt.legend()
    plt.title(title)
    if lim is not None:
        plt.xlim(lim)
        plt.ylim(lim)
    plt.show()


def plot_line_of_best_fit(x, y):
    # Compute best fit line (1st degree polynomial)
    slope, intercept = np.polyfit(x, y, 1)

    # Compute predicted values
    y_fit = slope * x + intercept
    r = np.corrcoef(x, y)[0, 1]

    # Plot
    plt.scatter(x, y, color='steelblue', label='Data', s = 2)
    plt.plot(x, y_fit, color='black', linewidth=2, label=f'Fit r={r:0.3f}') # Best fit: y={slope:.2f}x+{intercept:.2e}
    plt.legend()
    plt.show()
    return r