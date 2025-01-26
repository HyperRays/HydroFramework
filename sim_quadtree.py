# pylint: skip-file
import numpy as np
from config import Config
from os import path
from space import Space
from quadtree_helper import read_boundaries_from_file,read_nodes_from_file,read_sqmatrix_from_file

try:
    import torch
except:
    pass

class Quadtree(Space):
    """
    Quadtree dynamic resolution space
    """

    def setup(self, cfg: Config, file_path=None, use_torch=True):
        
        self.uses_torch = use_torch
        store = dict()

        if file_path is None:
            
            boundary = read_boundaries_from_file()
            depth,neighbors,store["virtual"] = read_nodes_from_file()
            store["sqmatrix_map"] = read_sqmatrix_from_file()

        elif isinstance(file_path, str):
            
            boundary = read_boundaries_from_file(path.join(file_path,"boundary_data.txt"))
            depth,neighbors,store["virtual"] = read_nodes_from_file(path.join(file_path,"nodes_data.txt"))
            store["sqmatrix_map"] = read_sqmatrix_from_file(path.join(file_path,"sqmatrix_data.txt"))
        
        elif isinstance(file_path, dict):

            boundary = read_boundaries_from_file(file_path["boundary"])
            depth,neighbors,store["virtual"] = read_nodes_from_file(file_path["nodes"])
            store["sqmatrix_map"] = read_sqmatrix_from_file(file_path["sqmap"])
        
        else:
            raise ValueError("Usage for file path is: folder_path[str], dict[boundary: boundary file, nodes: nodes file,sqmap: sqmap file]")
        
    
        neighbors = np.array(
            [
                neighbors[2],  # Up
                neighbors[3],  # Down
                neighbors[0],  # Left
                neighbors[1],  # Right
            ], dtype=int
        )

        store["depth"] = np.array(depth)
        dx = np.array([cfg.vol[0]/(2**store["depth"]),cfg.vol[1]/(2**store["depth"])])
        

        if use_torch:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.data_in_dir = self._data_in_dir_torch
            neighbors = [torch.from_numpy(n).long().to(device) for n in neighbors]

            # Convert store depth to PyTorch tensor
            store["depth"] = torch.from_numpy(store["depth"]).long().to(device)
            
            # Convert store virtual to list of PyTorch tensors
            store["virtual"] = [torch.from_numpy(v).long().to(device) for v in store["virtual"]]

        else:
            self.data_in_dir = self._data_in_dir_numpy

        return neighbors, boundary, dx, store
    
    def external_setup(self, cfg):
        return np.zeros((len(self.store["depth"]),))

    def from_grid(self, cfg):

        
        SqMatrix = np.array(
            [
                cfg.list_p,  # density
                cfg.list_p * cfg.list_v[0],  # momentum x
                cfg.list_p * cfg.list_v[1],  # momentum y
                (
                    cfg.list_e
                    + 0.5 * cfg.list_p * np.sum(cfg.list_v, 0) ** 2
                ),  # energy
            ]
        )

        assert self.store["sqmatrix_map"].shape == SqMatrix.shape[1:], f"Generated Quadtree requires shape {self.store['sqmatrix_map'].shape} given shape {SqMatrix.shape[1:]}"

        # use sqmap to map grid values to flat list
        arraylen = len(self.store["depth"])
        res = self.store["sqmatrix_map"].shape[0]
        values = np.zeros((4, arraylen))

        for y in range(res):
            for x in range(res):
                values[:,self.store["sqmatrix_map"][x,y]] += SqMatrix[:, x, y]
        
        return values

    
    def _data_in_dir_numpy(self, data, dir, dist=1):
        n = self.neighbors[dir]
        data_out = np.empty((4, n.shape[0]))

        mask_n_ge_0 = n >= 0
        mask_n_lt_0 = n < 0

        ### If values of the neighbors list are negative, they are pointing to a virtual node
        ### In this case use the abs index - 1 to get a nodes list from the virtual list
        ### then sum the values and pass on as 1 node (larger)
        ### use masks to decide when to access the virtual list and which values are created by indirection  

        if np.any(mask_n_ge_0):
            n_ge_0 = n[mask_n_ge_0]
            depth_self = self.store["depth"][mask_n_ge_0]
            depth_neighbor = self.store["depth"][n_ge_0]
            conv = 4 ** np.abs(depth_self - depth_neighbor)
            data_out[:, mask_n_ge_0] = data[:, n_ge_0] / conv

        if np.any(mask_n_lt_0):
            indices = -n[mask_n_lt_0] - 1
            if len(self.store["virtual"]) == 0:
                data_out[:, mask_n_lt_0] = 0
            else:
                unique_indices, inverse_indices = np.unique(indices, return_inverse=True)
                precomp_sums = np.empty((4, len(unique_indices)))

                for idx, virtual_idx in enumerate(unique_indices):
                    v = self.store["virtual"][virtual_idx]
                    precomp_sums[:, idx] = np.sum(data[:, v], axis=1)

                data_out[:, mask_n_lt_0] = precomp_sums[:, inverse_indices]

        return data_out

    def _data_in_dir_torch(self, data, dir, dist=1):
        n = self.neighbors[dir]
        data_out = torch.empty((4, n.shape[0]), dtype=data.dtype, device=data.device)

        mask_n_ge_0 = n >= 0
        mask_n_lt_0 = n < 0

        if torch.any(mask_n_ge_0):
            n_ge_0 = n[mask_n_ge_0]
            depth_self = self.store["depth"][mask_n_ge_0]
            depth_neighbor = self.store["depth"][n_ge_0]
            conv = 4 ** torch.abs(depth_self - depth_neighbor)
            data_out[:, mask_n_ge_0] = data[:, n_ge_0] / conv

        if torch.any(mask_n_lt_0):
            indices = -n[mask_n_lt_0] - 1
            if len(self.store["virtual"]) == 0:
                data_out[:, mask_n_lt_0] = 0
            else:
                unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
                precomp_sums = torch.empty((4, len(unique_indices)), dtype=data.dtype, device=data.device)

                unique_indices_list = unique_indices.tolist()
                for idx, virtual_idx in enumerate(unique_indices_list):
                    v = self.store["virtual"][virtual_idx]
                    precomp_sums[:, idx] = torch.sum(data[:, v], dim=1)

                data_out[:, mask_n_lt_0] = precomp_sums[:, inverse_indices]

        return data_out

    
    def to_grid(self, U):
        if self.uses_torch:
            maxd = np.max(self.store["depth"].cpu().numpy())
        else:
            maxd = np.max(self.store["depth"])
        idx = self.store["sqmatrix_map"]
        tmp = U[:, idx]/4**(maxd-self.store["depth"][idx])
        return tmp
     