import numpy as np


def tsp_nn_with_costs(distance_matrix):
    num_cities = distance_matrix.shape[0]
    all_paths = []
    all_costs = []

    for start_city in range(num_cities):
        visited = [False] * num_cities
        path = [start_city]
        total_cost = 0
        current_city = start_city
        visited[current_city] = True

        for _ in range(num_cities - 1):
            next_city = np.argmin(
                [
                    distance_matrix[current_city, j] if not visited[j] else np.inf
                    for j in range(num_cities)
                ]
            )
            path.append(next_city)
            visited[next_city] = True
            total_cost += distance_matrix[current_city, next_city]
            current_city = next_city

        total_cost += distance_matrix[current_city, start_city]  # Return to start city
        path.append(start_city)
        all_paths.append(path)
        all_costs.append(total_cost)

    return all_paths, all_costs


def print_path_with_costs(paths, costs):
    for i, (path, cost) in enumerate(zip(paths, costs)):
        print(f"Start from city {i}:")
        print(f"Path: {' -> '.join(map(str, path))}")
        print(
            f"Cost\t{' '.join(str(distance_matrix[path[j-1], path[j]]) for j in range(1, len(path)))}"
        )
        print(f"Path cost: {cost}\n")


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

paths, costs = tsp_nn_with_costs(distance_matrix)
print_path_with_costs(paths, costs)
