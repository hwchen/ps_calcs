# -*- coding: utf-8 -*-

''' Import statements '''
from .density import density

def distance(rcas, proximities):
  
  return 1 - density(rcas, proximities)

if __name__ == "__main__":
    import pandas as pd
    from rca import rca
    from proximity import proximity

    my_tbl = pd.DataFrame({ 0: [1,2], 1: [3,4], 2:[5,6] })
    print(my_tbl)

    my_rca = rca(my_tbl, None)
    print(my_rca)

    my_proximity = proximity(my_rca)
    print(my_proximity)

    my_distance = distance(my_rca, my_proximity)
    print(my_distance)
