import json
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import informed_rrt_star3D
import informed_rrv_star3D
time_points = np.arange(0, 40, 0.01)


def read_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def update_config(file_path, environment):
    config = read_config(file_path)
    config['env'] = environment
    with open(file_path, 'w') as file:
        json.dump(config, file)


# 筛选数据的函数
def filter_data(x, y):
    filtered_x = [nx for nx, ny in zip(x, y) if ny < 1000]
    filtered_y = [ny for ny in y if ny < 1000]
    return filtered_x, filtered_y


# 从原始代码中整合的 process_data 函数
def process_data(function, i):
    time_cost_list, time_success_list = function(i)
    time_cost_list.insert(0, [0, 1e18])
    time_success_list.insert(0, [0, 0])
    print("drawing list curve ")
    print(time_cost_list)
    print(time_success_list)
    time_cost_array = np.array(time_cost_list)
    time_success_array = np.array(time_success_list)
    f_cost = interp1d(time_cost_array[:, 0], time_cost_array[:, 1], kind='linear')
    f_success = interp1d(time_success_array[:, 0], time_success_array[:, 1], kind='linear')

    cost_values = f_cost(time_points)
    success_values = f_success(time_points)
    return cost_values, success_values


# 从原始代码中整合的 run 函数
def run(num_runs, function):
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cores * 2)
    results = pool.starmap(process_data, [(function, i) for i in range(num_runs)])
    pool.close()
    pool.join()
    cost_values = np.median(np.stack([result[0] for result in results], axis=0), axis=0)
    success_values = np.sum(np.stack([result[1] for result in results], axis=0), axis=0) / num_runs
    return time_points, cost_values, time_points, success_values


# 绘图函数
def draw_graph(ax, x, y, title, label):
    ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.legend()


def filter(x, y):
    x_1 = [nx for nx, ny in zip(x, y) if ny < 200]
    y_1 = [ny for ny in y if ny < 50]
    return x_1, y_1


def main():
    functions = [
        (informed_rrv_star3D.main, 'IRRV'),
        (informed_rrt_star3D.main, 'IRRT')
    ]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    print("drawing the 3D")
    num_runs = 8
    for func, label in functions:
        x_1, y_1, x_2, y_2 = run(num_runs, func)
        print("cost is ")
        print(y_1)
        filter(x_1, y_1)
        axs[0].set_xlim(0, 100)
        axs[1].set_xlim(0, 100)
        axs[0].set_ylim(0, 100)
        axs[1].set_ylim(0, 1)
        draw_graph(axs[0], x_1, y_1, f'Environment  - Cost vs. Time', label)
        draw_graph(axs[1], x_2, y_2, f'Environment  - Success Ratio vs. Time', label)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
