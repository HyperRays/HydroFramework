from config import Config

class Space:
    """
    Base class for simulation spaces
    """

    def __init__(self, cfg: Config, **kwargs): 
        
        self.cfg = cfg
        self.neighbors,self.boundary,self.dx,self.store = self.setup(cfg, **kwargs)
        self.values = self.from_grid(cfg)
        self.external = self.external_setup(cfg)


    def U_vector(self): return self.values
    def data_in_dir(self, data, dir): ...

    def setup(self,cfg) -> tuple: ...
    def external_setup(self): ...
    
    def from_grid(self, cfg): ...
    def to_grid(self,U): ...

