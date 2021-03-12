import numpy as np
import pandas as pd
import os, sys
import h5py
import time

# eztao
from eztao.ts import drw_fit, dho_fit

# dask
from dask.distributed import Client

bands = ["u", "g", "r", "i", "z", "y"]
bands_dict = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z':4, 'y':5} 

def fit_lc(hdf5_path, object_id):
    """Function to fit LC give the path to the hdf5 file and the object_id."""
    
    f = h5py.File(hdf5_path, 'r')
    
    # define some placeholders
    a1_temp = 'a1_{}'
    a2_temp = 'a2_{}'
    b0_temp = 'b0_{}'
    b1_temp = 'b1_{}'
    return_dict = {}
    
    # read lc from hdf5
    lc = pd.DataFrame.from_records(f[str(object_id)][:])

    # go through each passband
    for band in bands:
        
        try:
            # get lc in the specified band
            lc_band = lc[lc.passband == bands_dict[band]].sort_values('mjd').copy()

            # remove large errors
            max_err = np.percentile(lc_band.flux_err, 99)
            lc_band = lc_band[lc_band.flux_err < max_err]

            # get ts
            t = lc_band.mjd.values
            y = lc_band.flux.values
            yerr = lc_band.flux_err.values

            # fit
            best_dho = dho_fit(t, y, yerr, scipy_opt_kwargs={'tol': 1e-14}, n_iter=100)

            # assign best-fit to dict
            return_dict[a1_temp.format(band)] = best_dho[0]
            return_dict[a2_temp.format(band)] = best_dho[1]
            return_dict[b0_temp.format(band)] = best_dho[2]
            return_dict[b1_temp.format(band)] = best_dho[3]
        except:
            # assign best-fit to dict
            return_dict[a1_temp.format(band)] = np.nan
            return_dict[a2_temp.format(band)] = np.nan
            return_dict[b0_temp.format(band)] = np.nan
            return_dict[b1_temp.format(band)] = np.nan

    # add object_id to dict
    return_dict['object_id'] = object_id

    return return_dict



if __name__ == '__main__':
    
    start = time.time()
    client = Client(n_workers=4)
    
    # paths
    meta_path = sys.argv[1]
    lc_path = sys.argv[2]
    out_path = sys.argv[3]
    
    # read meta into df
    train_meta = pd.read_parquet(meta_path)
    
    rt = []
    # loop over all opsims to evaluate and submite task to dask scheduler
    for object_id in train_meta.object_id.values[:]:
        r = client.submit(fit_lc, lc_path, object_id)
        rt.append(r)

    # collect result
    result = client.gather(rt)
    
    # to df and save to disk
    result_df = pd.DataFrame(result)
    result_df.to_parquet(out_path)
    
    print((time.time() - start)/60)
    
    