import numpy as np

def read_nodes_from_file(file_name="nodes_data.txt"):
    with open(file_name, 'r') as file:
        # read lines from file
        lines = file.readlines()

        # first line depths
        depth_list = [int(depth) for depth in lines[0].split()]

        # next four lines neighbors
        neighbors_list = []
        for i in range(1, 5):  # four directions (left, right, up, down)
            direction_neighbors = [int(neighbor) for neighbor in lines[i].split()]
            neighbors_list.append(direction_neighbors)

        # remaining lines virtual nodes subarrays
        virtual_list = []
        for line in lines[5:]:
            virtual_sublist = np.array([int(v) for v in line.split()])
            virtual_list.append(virtual_sublist)

    return depth_list, neighbors_list, virtual_list


def read_sqmatrix_from_file(file_name="sqmatrix_data.txt"):
    # Initialize an empty list to store rows
    sqmatrix = []

    # Open file and read its contents
    with open(file_name, 'r') as file:
        for line in file:
            # Split line into individual values and convert them to integers
            row = [int(value) for value in line.split()]
            # Append row to sqmatrix
            sqmatrix.append(row)

    return np.array(sqmatrix).T


def read_boundaries_from_file(file_name="boundary_data.txt"):
    with open(file_name, 'r') as file:
        boundary = [int(value) for value in file.read().split()]
    
    return boundary
