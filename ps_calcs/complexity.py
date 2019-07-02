# -*- coding: utf-8 -*-

''' Import statements '''
import sys
import numpy as np

def complexity(rcas, drop=True):
  
  rcas_clone = rcas.copy()
  
  # drop columns / rows only if completely nan
  rcas_clone = rcas_clone.dropna(how="all")
  rcas_clone = rcas_clone.dropna(how="all", axis=1)
  
  if rcas_clone.shape != rcas.shape:
    print("[Warning] RCAs contain columns or rows that are entirely comprised of NaN values.")
    if drop:
      rcas = rcas_clone
  
  kp = rcas.sum(axis=0)
  kc = rcas.sum(axis=1)
  kp0 = kp.copy()
  kc0 = kc.copy()

#  print("kp0\n:", kp0)
#  print("kc0\n:", kc0)

  for i in range(1, 20):
    kc_temp = kc.copy()
    kp_temp = kp.copy()
    kp = rcas.T.dot(kc_temp) / kp0
    if i < 19:
      kc = rcas.dot(kp_temp) / kc0

#  print("kp:\n", kp)
#  print("kc:\n", kc)

#  print("kp_mean:\n", kp.mean())
#  print("kc_mean:\n", kc.mean())
  
  print("kp:", kp, "kp_std:", kp.std())
#  print("kc_std:\n", kc.std())
  
  geo_complexity = (kc - kc.mean()) / kc.std()
  prod_complexity = (kp - kp.mean()) / kp.std()

  return geo_complexity, prod_complexity

if __name__ == "__main__":
    import pandas as pd
    from rca import rca

    my_tbl = pd.DataFrame({ 0: [100,2000], 1: [3,4000], 2:[500,6000], 3: [17, 23] })
#    print(my_tbl)

    my_rca = rca(my_tbl, None)
#    print(my_rca)

    my_complexity = complexity(my_rca)
#    print(my_complexity)
