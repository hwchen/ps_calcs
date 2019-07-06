# -*- coding: utf-8 -*-

''' Import statements '''
import sys
import numpy as np
import pandas as pd

def density(rcas, proximities):

  # Get numerator by matrix multiplication of proximities with M_im
  density_numerator = rcas.dot(proximities)
  #print("density_numerator:\n", density_numerator)

  # Get denominator by multiplying proximities by all ones vector thus
  # getting the sum of all proximities
  # rcas_ones = pd.DataFrame(np.ones_like(rcas))
  rcas_ones = rcas * 0
  #print("rcas_ones:\n", rcas_ones)
  rcas_ones = rcas_ones + 1
  #print("rcas_ones:\n", rcas_ones)
  # print rcas_ones.shape, proximities.shape 
  density_denominator = rcas_ones.dot(proximities)
  
  # We now have our densities matrix by dividing numerator by denomiator
  densities = density_numerator / density_denominator

  return densities

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

    my_density = density(my_rca, my_proximity)
    print(my_density)

    # zeros and ones
    my_tbl = pd.DataFrame({ 0: [0,1], 1: [1,0], 2:[0,1] })
    print(my_tbl)

    my_rca = rca(my_tbl, None)
    print(my_rca)

    my_proximity = proximity(my_rca)
    print(my_proximity)

    my_density = density(my_rca, my_proximity)
    print(my_density)
