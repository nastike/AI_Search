import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Given distance matrix
distance_matrix = np.array(
    [
        [0, 10, 10, 12, 5, 4, 19, 14, 3, 18],
        [10, 0, 23, 45, 27, 4, 24, 9, 34, 12],
        [10, 23, 0, 34, 22, 45, 20, 23, 22, 21],
        [12, 45, 34, 0, 12, 12, 26, 43, 33, 21],
        [5, 27, 22, 12, 0, 13, 21, 5, 7, 22],
        [4, 45, 12, 13, 25, 0, 11, 22, 12, 10],
        [19, 24, 20, 26, 21, 11, 0, 20, 33, 11],
        [14, 9, 23, 43, 5, 22, 20, 0, 6, 23],
        [3, 34, 22, 33, 7, 12, 33, 6, 0, 17],
        [18, 12, 21, 21, 22, 10, 11, 23, 17, 0],
    ]
)


def calculate_path_cost(path, distance_matrix):
    total_cost = 0
    for i in range(len(path) - 1):
        total_cost += distance_matrix[path[i]][path[i + 1]]
    total_cost += distance_matrix[path[-1]][path[0]]  # Return to starting city
    return total_cost


def two_opt(path, distance_matrix):
    best_path = path[:]
    improved = True
    iterations = []
    while improved:
        improved = False
        for i in range(1, len(best_path) - 1):
            for j in range(i + 1, len(best_path)):
                new_path = best_path[:i] + best_path[i:j][::-1] + best_path[j:]
                new_cost = calculate_path_cost(new_path, distance_matrix)
                swapped_cities = f"{best_path[i]+1},{best_path[j]+1}"  # Convert to 1-indexed and format as string
                iterations.append(
                    {
                        "Path": ",".join(
                            map(str, [city + 1 for city in new_path])
                        ),  # Convert to 1-indexed and format as string
                        "Total Cost": int(new_cost),
                        "Swapped": swapped_cities,  # Store the swapped cities as string
                    }
                )
                if new_cost < calculate_path_cost(best_path, distance_matrix):
                    best_path = new_path
                    improved = True
    return best_path, calculate_path_cost(best_path, distance_matrix), iterations


def nearest_neighbor(distance_matrix, start_city=0):
    num_cities = distance_matrix.shape[0]
    visited = [False] * num_cities
    path = [start_city]
    visited[start_city] = True
    current_city = start_city
    total_cost = 0

    for _ in range(num_cities - 1):
        min_distance = np.inf
        nearest_city = None
        for next_city in range(num_cities):
            if (
                not visited[next_city]
                and distance_matrix[current_city][next_city] < min_distance
            ):
                min_distance = distance_matrix[current_city][next_city]
                nearest_city = next_city
        path.append(nearest_city)
        visited[nearest_city] = True
        total_cost += min_distance
        current_city = nearest_city

    total_cost += distance_matrix[path[-1]][start_city]  # Return to starting city

    # Format path as string
    path_str = ",".join(map(str, [city + 1 for city in path]))

    return (
        path,
        total_cost,
        [{"Path": path_str, "Total Cost": int(total_cost), "Swapped": ""}],
    )


def main():
    # Apply Nearest Neighbor algorithm
    best_path_nn = None
    best_cost_nn = np.inf
    iterations_nn = []
    for start_city in range(len(distance_matrix)):
        final_path, path_cost, nn_iterations = nearest_neighbor(
            distance_matrix, start_city
        )
        iterations_nn.extend(nn_iterations)
        if path_cost < best_cost_nn:
            best_path_nn = final_path
            best_cost_nn = path_cost

    # Apply 2-opt method
    best_path_2opt, best_cost_2opt, iterations_2opt = two_opt(
        best_path_nn, distance_matrix
    )

    # Print optimized cost and path
    print("Nearest Neighbor Optimized Cost:", best_cost_nn)
    print(
        "Nearest Neighbor Optimized Path:",
        "->".join(map(str, [city + 1 for city in best_path_nn])),
    )

    print("2-opt Optimized Cost:", best_cost_2opt)
    print(
        "2-opt Optimized Path:",
        "->".join(map(str, [city + 1 for city in best_path_2opt])),
    )

    # Export iterations to CSV
    df_nn = pd.DataFrame(iterations_nn)
    df_nn.to_csv("tsp_iterations_nn.csv", index=False)

    df_2opt = pd.DataFrame(iterations_2opt)
    df_2opt.to_csv("tsp_iterations_2opt.csv", index=False)

    # Create a graph showing distance against iterations for 2-opt method
    iterations_data_2opt = [
        (i + 1, iteration["Total Cost"]) for i, iteration in enumerate(iterations_2opt)
    ]
    iterations_2opt, total_costs_2opt = zip(*iterations_data_2opt)

    plt.figure(figsize=(10, 6))
    plt.plot(iterations_2opt, total_costs_2opt, marker="o", linestyle="-")
    plt.xlabel("Iterations")
    plt.ylabel("Total Cost")
    plt.title("Total Cost of Path vs. Iterations (2-opt)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
