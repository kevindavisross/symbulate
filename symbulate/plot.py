import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import itertools
from scipy.stats import gaussian_kde
from cycler import cycler

figure = plt.figure

xlabel = plt.xlabel
ylabel = plt.ylabel

xlim = plt.xlim
ylim = plt.ylim

color_index=0
color_cycle = [c['color'] for c in plt.rcParams['axes.prop_cycle']]

def init_color():
    hex_list = [colors.rgb2hex(rgb) for rgb in cm.get_cmap('tab10').colors]
    plt.rcParams["axes.prop_cycle"] = cycler('color', hex_list)

def get_next_color(axes):
    if not hasattr(axes, "_color_cycle"):
        prop_cycle = plt.rcParams['axes.prop_cycle']
        axes._color_cycle = itertools.cycle(prop_cycle.by_key()['color'])
    return next(axes._color_cycle)

def configure_axes(axes, xdata, ydata, xlabel = None, ylabel = None):
    # Create 5% buffer on either end of plot so that leftmost and rightmost
    # lines are visible. However, if current axes are already bigger,
    # keep current axes.
    buff = .05 * (max(xdata) - min(xdata))
    xmin, xmax = axes.get_xlim()
    xmin = min(xmin, min(xdata) - buff)
    xmax = max(xmax, max(xdata) + buff)
    plt.xlim(xmin, xmax)

    _, ymax = axes.get_ylim()
    ymax = max(ymax, 1.05 * max(ydata))
    plt.ylim(0, ymax)
    
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

def plot(*args, **kwargs):
    try:
        args[0].plot(**kwargs)
    except:
        plt.plot(*args, **kwargs)
        
def is_discrete(heights):
    return sum([(i > 1) for i in heights]) > .8 * len(heights)

def count_var(x):
    counts = {}
    for val in x:
        if val in counts:
            counts[val] += 1
        else:
            counts[val] = 1
    return counts
    
def compute_density(values):
    density = gaussian_kde(values)
    density.covariance_factor = lambda: 0.25
    density._compute_covariance()
    return density

def setup_ticks(pos, lab, ax):
    ax.set_ticks(pos)
    ax.set_ticklabels(lab)

def reduce_ticks(x_shape, y_shape, ax):
    # Plot mutating function
    # x_shape = # of discrete x values, y_shape = # of discrete y values
    # Use this if there is a discrete RV in the 2D sim results plot

    # Here we initialize the list of ticks and their labels that we want to keep
    xticks, xlabels, yticks, ylabels = [], [], [], []
    # If the number of discrete x values is too much...
    if x_shape > 20:
        # First we get the tick labels (label) and their x location
        for x, label in enumerate(ax.xaxis.get_ticklabels()):
            # We filter down to 20 ticks and their labels by adding 20 evenly spaced ticks/labels to the new lists
            if x % (x_shape // 20) == 0:
                xticks.append(label.get_position()[0])
                xlabels.append(str(round(float(label.get_text()))))
        # Then we plot them
        plt.xticks(xticks, labels=xlabels, rotation=75)
    # Repeat for y-axis
    if y_shape > 15:
        for y, label in enumerate(ax.yaxis.get_ticklabels()):
            if y % (y_shape // 15) == 0:
                yticks.append(label.get_position()[1])
                ylabels.append(str(round(float(label.get_text()))))
        plt.yticks(yticks, labels=ylabels)
    
def add_colorbar(fig, type, mappable, label):
    #create axis for cbar to place on left
    if 'marginal' not in type: 
        caxes = fig.add_axes([0, 0.1, 0.05, 0.8])
    else: #adjust height if marginals
        caxes = fig.add_axes([0, 0.1, 0.05, 0.57])
    cbar = plt.colorbar(mappable=mappable, cax=caxes)
    caxes.yaxis.set_ticks_position('left')
    cbar.set_label(label)
    caxes.yaxis.set_label_position('left')
    return caxes

def setup_tile(v, bins, discrete):
    if not discrete:
        v_lab = np.linspace(min(v), max(v), bins + 1)
        v_pos = np.arange(0, len(v_lab)) - 0.5
        v_vect = np.digitize(v, v_lab, right=True) - 1
    else:
        v_lab = np.unique(v) #returns sorted array
        v_pos = range(len(v_lab))
        v_map = dict(zip(v_lab, v_pos))
        v_vect = np.vectorize(v_map.get)(v)
    return v_vect, v_lab, v_pos

def make_tile(x, y, bins, discrete_x, discrete_y, ax):
    x_vect, x_lab, x_pos = setup_tile(x, bins, discrete_x)
    y_vect, y_lab, y_pos = setup_tile(y, bins, discrete_y)
    nums = len(x_vect)
    counts = count_var(list(zip(y_vect, x_vect)))
    y_shape = len(y_lab) if discrete_y else len(y_lab) - 1
    x_shape = len(x_lab) if discrete_x else len(x_lab) - 1
    intensity = np.zeros(shape=(y_shape, x_shape))
        
    for key, val in counts.items():
        intensity[key] = val / nums
    if not discrete_x: x_lab = np.around(x_lab, decimals=1)
    if not discrete_y: y_lab = np.around(y_lab, decimals=1)
    hm = ax.matshow(intensity, cmap='viridis', origin='lower', aspect='auto', vmin=0)
    ax.xaxis.set_ticks_position('bottom')
    setup_ticks(x_pos, x_lab, ax.xaxis)
    setup_ticks(y_pos, y_lab, ax.yaxis)
    reduce_ticks(x_shape, y_shape, ax)

    return hm

def make_violin(data, positions, ax, axis, alpha):
    values = []
    i, j = (0, 1) if axis == 'x' else (1, 0)
    values = [data[data[:, i] == pos, j].tolist() for pos in positions]
    violins = ax.violinplot(dataset=values, showmedians=True,
                            vert=False if axis == 'y' else True)
    setup_ticks(np.array(positions) + 1, positions, 
                ax.xaxis if axis == 'x' else ax.yaxis)
    for part in violins['bodies']:
        part.set_edgecolor('black')
        part.set_alpha(alpha)
    for component in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = violins[component]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)

def make_marginal_impulse(count, color, ax_marg, alpha, axis):
    key, val = list(count.keys()), list(count.values())
    tot = sum(val)
    val = [i / tot for i in val]
    if axis == 'x':
        ax_marg.vlines(key, 0, val, color=color, alpha=alpha)
    elif axis == 'y':
        ax_marg.hlines(key, 0, val, color=color, alpha=alpha)

def make_density2D(x, y, ax):
    res = np.vstack([x, y])
    density = gaussian_kde(res)
    xmax, xmin = max(x), min(x)
    ymax, ymin = max(y), min(y)
    Xgrid, Ygrid = np.meshgrid(np.linspace(xmin, xmax, 100),
                               np.linspace(ymin, ymax, 100))
    Z = density.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
    den = ax.imshow(Z.reshape(Xgrid.shape), origin='lower', cmap='viridis',
              aspect='auto', extent=[xmin, xmax, ymin, ymax]
    )
    return den

def make_mosaic(data, ax):
    # data is a list of [x, y] pairs
    # Split data into x and y values
    x, y = data[:, 0], data[:, 1]
    # Get number of simulated values
    data_len = len(data)

    # Get unique values of y
    y_vals = list(set(y))
    # Get marginal distribution of y as a list, using the counts of y values
    y_marg = [i / data_len for i in dict(sorted(count_var(y).items())).values()]
    # Create y-coordinates for the bar chart squares based on the marginal distribution
    y_loc, loc = [0], 0
    for j in y_marg[:-1]:
        loc += j
        y_loc.append(loc)

    # Create a vector to map color values to support by rescaling btwn 0 to 1
    y_colors = [(i - min(y_vals)) / (max(y_vals) - min(y_vals)) for i in y_vals]

    # Get unique values of x
    x_vals = list(set(x))
    # Get marginal distribution of x as a list, using the counts of x values
    x_marg = [i / data_len for i in dict(sorted(count_var(x).items())).values()]
    # Create x-coordinates for the bar chart squares based on the marginal distribution
    x_loc, loc = [0], 0
    for j in x_marg[:-1]:
        loc += j
        x_loc.append(loc)
    # Get number of unique values of x
    num_x = len(x_vals)

    # Initialize the matrix of counts
    x_counts = []

    # Create the matrix of counts as a list of lists
    # Each list/"row" is a y-value, therefore each "column" is an x-value
    for i in y_vals:
        x_given_y = np.array([val[0] for val in data if val[1] == i])
        x_y_count = count_var(x_given_y)
        # Include x-values with a count of 0
        for val in x_vals:
            if val not in x_y_count.keys():
                x_y_count[val] = 0
        x_y_count = dict(sorted(x_y_count.items()))
        x_y_values = list(x_y_count.values())
        x_counts.append(x_y_values)

    # Turn the matrix of counts into a numpy array
    # This way, we can use numpy methods
    count_matrix = np.array(x_counts)
    # Get the marginal count across each "column" (each value of x)
    y_sum = np.sum(count_matrix, axis=0)
    # Get a matrix where each element is the conditional probability of y | x
    y_prop_matrix = np.divide(count_matrix, y_sum)


    # Initialize whitespace, which will be used to determine the
    #   whitespace below each square on the mosaic plot
    #   by summing the height of each of the squares below it
    whitespace = np.zeros(num_x)

    # Create a new bar plot with a new color for each y-value.
    # As you can see in the resulting plot, each color is actually
    #   an individual plot that has been layered on.
    # Each individual plot/color represents a single y-value
    for r in range(len(y_prop_matrix)):
        row = y_prop_matrix[r]
        # Base the color on the y-value
        color_val = y_colors[r]
        # Create the plot
        plt.bar(x_loc, row, bottom=whitespace, width=x_marg,
                align='edge', edgecolor="white", color=get_viridis(color_val))
        # Add the height of the current y-values to the whitespace for the next plot
        whitespace = whitespace + np.array(row)

    plt.ylim((0, 1))
    plt.xlim((0, 1))

    # Create a new axis on the top of the plot that show the
    #   x-values at marginal positions instead of proportions
    # axy = ax.twiny()
    ax.set_xticks([(x_marg[i] / 2) + v for i, v in enumerate(x_loc)])
    ax.set_xticklabels(x_vals)
    ax.twiny()

def make_mosaic_marginal(data, ax):
    y = data[:, 1]
    # Get number of simulated values
    data_len = len(data)

    # Get unique values of y
    y_vals = list(set(y))
    # Get marginal distribution of y as a list, using the counts of y values
    y_marg = [i / data_len for i in dict(sorted(count_var(y).items())).values()]
    # Create y-coordinates for the bar chart squares based on the marginal distribution
    y_loc, loc = [0], 0
    for j in y_marg[:-1]:
        loc += j
        y_loc.append(loc)

    y_colors = [(i - min(y_vals)) / (max(y_vals) - min(y_vals)) for i in y_vals]

    whitespace = 0

    for r in range(len(y_marg)):
        row = y_marg[r]
        # Base the color on the y-value
        color_val = y_colors[r]
        # Create the plot
        plt.bar(0, row, bottom=whitespace, width=1,
                align='edge', edgecolor="white", color=get_viridis(color_val))
        # Add the height of the current y-values to the whitespace for the next plot
        whitespace = whitespace + row

    plt.ylim((0, 1))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['', ''])

    # Create a new axis at the right of the plot that shows the
    #   y-values at marginal positions
    ax.set_yticks([(y_marg[i] / 2) + v for i, v in enumerate(y_loc)])
    ax.set_yticklabels(y_vals)
    ax.yaxis.tick_right()

def get_viridis(prop):
    cmap = plt.get_cmap('viridis')
    rgba = cmap(prop)
    return colors.rgb2hex(rgba)

