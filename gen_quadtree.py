import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class TreeNode:
    def __init__(self, x, y, width, height, depth=0, min_depth=10, max_depth=15, parent=None, depth_formula=None):
        self.x = x
        self.y = y
        self.center_x = x + (width / 2)
        self.center_y = y + (height / 2)
        self.width = width
        self.height = height
        self.depth = depth
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.children = []
        self.is_divided = False
        self.parent = parent
        self.depth_formula = depth_formula

        self.subdivide()

    def subdivide(self):
        # subdivide based on proximity to center
        if self.depth < self.get_effective_max_depth():
            half_width = self.width / 2
            half_height = self.height / 2

            self.children = [
                TreeNode(self.x, self.y, half_width, half_height, self.depth + 1, self.min_depth, self.max_depth, self, self.depth_formula),  # top-left
                TreeNode(self.x + half_width, self.y, half_width, half_height, self.depth + 1, self.min_depth, self.max_depth, self, self.depth_formula),  # top-right
                TreeNode(self.x, self.y + half_height, half_width, half_height, self.depth + 1, self.min_depth, self.max_depth, self, self.depth_formula),  # bottom-left
                TreeNode(self.x + half_width, self.y + half_height, half_width, half_height, self.depth + 1, self.min_depth, self.max_depth, self, self.depth_formula),  # bottom-right
            ]

            self.is_divided = True

    def get_effective_max_depth(self):
        if self.depth_formula:
            # Use formula provided, which takes x, y, and max_depth as inputs
            return max(
                self.depth_formula(self.center_x       , self.center_y      , self.max_depth, self.min_depth),
                self.depth_formula(self.x              , self.y             , self.max_depth, self.min_depth),
                self.depth_formula(self.x + self.height, self.y             , self.max_depth, self.min_depth),
                self.depth_formula(self.x              , self.y + self.width, self.max_depth, self.min_depth),
                self.depth_formula(self.x + self.height, self.y + self.width, self.max_depth, self.min_depth),
            )
        else:
            # default is proximity to center
            center_x = width / 2  # system center
            center_y = height / 2

            dist_to_center = ((self.center_x - center_x) ** 2 + (self.center_y - center_y) ** 2) ** 0.5
            max_dist = (center_x**2 + center_y**2) ** 0.5
            normalized_dist = dist_to_center / max_dist

            return int(self.min_depth + (self.max_depth - self.min_depth) * (1 - normalized_dist))


    def draw(self, ax, highlight_cells=None, color='r'):
        # Draw node; highlight if it's in list of highlight_cells
        if highlight_cells and self in highlight_cells:
            rect = patches.Rectangle((self.x, self.y), self.width, self.height, linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.5)
            ax.add_patch(rect)
        
        if not self.is_divided:
            rect = patches.Rectangle((self.x, self.y), self.width, self.height, linewidth=1, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
        else:
            for child in self.children:
                child.draw(ax, highlight_cells, color)


    def find_max_depth(self):
        maxDepth = self.depth
        if not self.children:
            return maxDepth

        for child in self.children:
            child_maxDepth = child.find_max_depth()
            if child_maxDepth > maxDepth:
                maxDepth = child_maxDepth

        return maxDepth

    def get_neighbors(self, tree):
        neighbors = [None, None, None, None]  # left, right, up, down
        # define neighbor positions
        neighbor_positions = [
            [self.center_x - self.width, self.center_y],  # left
            [self.center_x + self.width, self.center_y],  # right
            [self.center_x, self.center_y + self.height],  # up
            [self.center_x, self.center_y - self.height],  # down
        ]

        # find neighbor nodes using find_point()
        for index, position in enumerate(neighbor_positions):
            x = position[0] % tree.root.width
            y = position[1] % tree.root.height
            neighbor_node, _ = tree.find_point([x, y], self.depth)
            neighbors[index] = neighbor_node

        return neighbors


class Tree:
    def __init__(self, width, height, min_depth=10, max_depth=15, depth_formula=None):
        self.depth_formula = depth_formula
        self.root = TreeNode(0, 0, width, height, 0, min_depth, max_depth, depth_formula=self.depth_formula)
        self.max_depth = max_depth

    # DISCLAIMER: ALL DRAWING LOGIC WAS DONE BY CHATGPT
    def draw(self, highlight_cells=None):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.root.width)
        ax.set_ylim(0, self.root.height)
        ax.set_aspect("equal")

        if highlight_cells is not None:
            self.root.draw(ax, set(highlight_cells))
        else:
            self.root.draw(ax, highlight_cells)
            
        # plt.gca().invert_yaxis()  # Optional: Invert y-axis to match common graphics coordinate system
        plt.show()

    def find_point(self, position, nodeDepth=None):
        if nodeDepth is None:
            nodeDepth = self.max_depth
        currentNode = self.root
        c_pos = position
        for _ in range(nodeDepth):  # search for node depth
            if not currentNode.is_divided:  # end node found
                break
            
            right = int(c_pos[0] > 0.5)  # either 1 or 0
            up = int(c_pos[1] > 0.5)
            index = (up << 1) | right  # bit wise operator using OR --> turning 0, 01, 10 and 11 into index
            currentNode = currentNode.children[index]  # get current nodes child in found quadrant
            c_pos = (c_pos[0]*2 - right, c_pos[1]*2 - up)
            
        maxDepth = currentNode.find_max_depth()
        return currentNode, maxDepth

    def generate_all_nodes(self):
        child_l = []
        virtual = []
        neighbors = [
            [],  # left
            [],  # right
            [],  # up
            [],  # down
        ]

        # helper function: recursive search for leaf nodes
        def child_func(node):
            temp = []
            for child in node.children:
                if child.is_divided:
                    temp += child_func(child)  # recursion
                else:
                    temp.append(child)  # leaf node
            return temp

        # takes array of leafs and returns each index in global list
        def leaf_to_index(all_n_map, lst):
            l_i = []
            for n in lst:
                l_i.append(all_n_map[n])
            return l_i

        child_l = child_func(self.root)  # initiate recursion
        child_l_map = dict(zip(child_l,range(len(child_l))))

        # loop through all leaf nodes to get pointers to neighbors
        for child in child_l:  
            neighbors_c = child.get_neighbors(self)
            for dir_lst_idx in range(len(neighbors)):
                if neighbors_c[dir_lst_idx] in child_l_map:
                    neighbors[dir_lst_idx].append(child_l_map[neighbors_c[dir_lst_idx]])
                else:
                    # get all of leaf nodes of a neighbor in a direction
                    neighbor_leaf_nodes = leaf_to_index(
                        child_l_map, child_func(neighbors_c[dir_lst_idx])
                    )
                    # add that list of leaf nodes to virtual nodes list
                    virtual.append(neighbor_leaf_nodes)
                    # append negative index + 1 of virtual node (-(index + 1))
                    neighbors[dir_lst_idx].append(-len(virtual))

        return child_l, neighbors, virtual

# WRITTEN BY CHATGPT USING EXAMPLES
def find_neighbors_by_direction(node_index, direction, c_l, neighbors, virtual):

    # Direction mappings
    direction_map = {"left": 0, "right": 1, "up": 2, "down": 3}

    # Get neighbor index in requested direction
    dir_idx = direction_map[direction]
    neighbor_idx = neighbors[dir_idx][node_index]

    # If neighbor index is positive, it's a direct neighbor from c_l
    if neighbor_idx >= 0:
        return [c_l[neighbor_idx]]

    # If neighbor index is negative, it's a virtual neighbor
    # absolute value of neighbor_idx (after negating) corresponds to index in virtual
    virtual_node_idx = -neighbor_idx - 1
    virtual_neighbors = virtual[virtual_node_idx]

    # Return corresponding smaller cells from c_l that make up virtual neighbor
    return [c_l[i] for i in virtual_neighbors]


def write_nodes_to_file(leaf_list, neighbors_list, virtual_list, file_name="nodes_data.txt"):
    depths = np.array([leaf.depth for leaf in leaf_list])
    neighbors_array = np.array(neighbors_list)
    virtual_array = np.array([np.array(v) for v in virtual_list], dtype=object)

    with open(file_name, 'w') as file:
        np.savetxt(file, depths.reshape(1, -1), fmt='%d')
        np.savetxt(file, neighbors_array, fmt='%d')
        for virtual_sublist in virtual_array:
            file.write(" ".join(map(str, virtual_sublist)) + "\n")

def map_sqmatrix_to_quadtree(max_resolution, tree, c_l):

    sqmatrix_to_leaf_node_map = np.ones((max_resolution,max_resolution))*-1
    cell_size = 1/max_resolution
    c_l_map = dict(zip(c_l,range(len(c_l))))

    for y in range(max_resolution):
        for x in range(max_resolution):
            position = [
                cell_size * (x+1) - 0.5 * cell_size,
                cell_size * (y+1) - 0.5 * cell_size,
            ]
            leaf_node_index = c_l_map[tree.find_point(position)[0]]
            sqmatrix_to_leaf_node_map[x,y] = leaf_node_index

    return sqmatrix_to_leaf_node_map

def write_sqmatrix_to_file(tree, leaf_list, max_resolution, file_name="sqmatrix_data.txt"):
    sqmatrix = map_sqmatrix_to_quadtree(max_resolution, tree, leaf_list)
    # with open(file_name, 'w') as file:
    #     file.write(np.array2string(sqmatrix,separator=" ",))
    np.savetxt(file_name, sqmatrix, fmt='%d')


def find_boundary_leaf_nodes(tree,c_l,max_resolution):
    
    sqmatrix_boundary = []
    cell_size = 1/max_resolution
    tree_boundary = set()
    c_l_map = dict(zip(c_l,range(len(c_l))))
    
    # Find boundaries for a sqmatrix 
    for x in range(max_resolution):
        for y in [0, max_resolution - 1]:
            sqmatrix_boundary += [(x, y)]

    for y in range(max_resolution):
        for x in [0, max_resolution - 1]:
            sqmatrix_boundary += [(x, y)]

    #find which nodes contain sqmatrix boundary cells (without duplicates)
    for (x,y) in sqmatrix_boundary:
        position = (
            cell_size * (x+1) - 0.5 * cell_size,
            cell_size * (y+1) - 0.5 * cell_size,
        )
        leaf_node_index = c_l_map[tree.find_point(list(position))[0]]
        tree_boundary.add(leaf_node_index)

    return list(tree_boundary)
            
def write_boundaries_to_file(qt, c_l, max_resolution, file_name="boundary_data.txt"):
    boundaries = find_boundary_leaf_nodes(qt, c_l, max_resolution)
    np.savetxt(file_name, np.array(boundaries).reshape(1, -1), fmt='%d')
