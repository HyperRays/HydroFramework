# pylint: skip-file
import numpy as np
from config import Config
from os import path
from space import Space
from quadtree_helper import read_boundaries_from_file,read_nodes_from_file,read_sqmatrix_from_file

class Quadtree(Space):
    """
    Quadtree dynamic resolution space
    """

    def setup(self, cfg: Config, file_path=None):
        
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

    
    def data_in_dir(self, data, dir): 
        
        if len(self.store["virtual"]) == 0:
            precomp_sum = np.array([0])
        else:
            precomp_sum = np.array([np.einsum('ij->i',data[:,v]) for v in self.store["virtual"]]).T
        
        n = self.neighbors[dir]
        conv = 4**(np.abs(self.store["depth"]-self.store["depth"][n]))

        data = np.select(
            [n < 0, n >= 0],
            [precomp_sum[:,np.maximum(-n-1,0)],data[:,n]/conv]
        )
        
        return data
    
    def to_grid(self, U):
        maxd = np.max(self.store["depth"])
        idx = self.store["sqmatrix_map"]
        tmp = U[:, idx]/4**(maxd-self.store["depth"][idx])
        return tmp
     