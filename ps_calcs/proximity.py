# -*- coding: utf-8 -*-

''' Import statements '''
import sys
import numpy as np

def proximity(rcas):
  
  # transpose the matrix so that it is now industries as rows and munics as
  # columns
  rcas_t = rcas.T.fillna(0)
  #rint("rcas_t:\n", rcas_t)
  #rint("rcas_t_t:\n", rcas_t.T)

  # Matrix multiplication on M_mi matrix and transposed version,
  # number of products = number of rows and vice versa on transposed
  # version, thus the shape of this result will be length of products by
  # by the length of products (symetric)
  numerator_intersection = rcas_t.dot(rcas_t.T)
  #rint("num_inter:\n", numerator_intersection)

  # kp0 is a vector of the number of munics with RCA in the given product
  kp0 = rcas.sum(axis=0)
  #print("kp0:\n", kp0)
  kp0 = kp0.values.reshape((1, len(kp0)))
  #print("kp0:\n", kp0)

  # transpose this to get the unions
  kp0_trans = kp0.T
  
  # multiply these two vectors, take the squre root
  # and then we have the denominator
  # denominator_union = kp0_trans.dot(kp0)
  denominator_union = kp0_trans.dot(kp0)

  # get square root for geometric mean
  denominator_union_sqrt = np.power(denominator_union, .5)
  #print("denominator_union_sqrt:\n", denominator_union_sqrt)

  # to get the proximities it is now a simple division of the untion sqrt
  # with the numerator intersections
  phi = np.divide(numerator_intersection, denominator_union_sqrt)
  
  return phi

if __name__ == "__main__":
    import pandas as pd
    from rca import rca

    my_tbl = pd.DataFrame({ 0: [1,2], 1: [3,4], 2:[5,6] })
    print(my_tbl)

    my_rca = rca(my_tbl, None)
    print(my_rca)

    my_proximity = proximity(my_rca)
    print(my_proximity)

    # with zero and 
    my_tbl = pd.DataFrame({ 0: [0,1], 1: [1,0], 2:[0,1] })
    print(my_tbl)

    my_rca = rca(my_tbl, None)
    print(my_rca)

    my_proximity = proximity(my_rca)
    print(my_proximity)
