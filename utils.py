import matplotlib.pyplot as plt
import numpy as np

def plot_contour(f,t1,t2,path_hist):
    # Create a grid of points
    x = np.linspace(t1[0], t1[1], 500)
    y = np.linspace(t2[0], t2[1], 500)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    # Compute function vals
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f.evaluate(np.array([X[i, j], Y[i, j]]))
    # Create the contour plot
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=100, cmap='viridis')

    # #define range for function to be displayed
    # min_y=10^6
    # min_x=10^6
    # max_y=-10^6
    # max_x=-10^6
    # Plot algorithm paths if provided
    if path_hist is not None:
        for name,hist in path_hist:
            _,x_values, y_values = zip(*hist)
            # if np.min(x_values)<min_x:
            #     min_x=np.min(x_values)
            # if np.min(y_values)<min_y:
            #     min_y=np.min(y_values)
            # if np.max(x_values)>max_x:
            #     max_x=np.max(x_values)
            # if np.max(y_values)>max_y:
            #     max_y=np.max(y_values)
            plt.plot(x_values, y_values,label=name, marker='.')
        plt.legend()
    #align with view
    # plt.ylim(min_y,max_y)
    # plt.xlim(min_x,max_x)
    # Set the title
    plt.title("Contour plot of the function and paths of all methods")
    plt.show()


def plot_function_values(f_vs):
    plt.figure(figsize=(10, 8))
    for method, function_values in f_vs.items():
        plt.plot(function_values, label=method, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Function value')
    plt.title('function values at each iteration for different methods')
    plt.legend()
    plt.grid(True)
    plt.show()