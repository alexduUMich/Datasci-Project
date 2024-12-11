import pandas as pd

import os

def list_all_files(directory):
    # Loop over all directories and files in the given directory
    paths = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            # Full path of the file
            full_path = os.path.join(dirpath, filename)
            if '.feather' in filename:
                paths.append(full_path)
    return paths

# Example usage:
directory_path = 'open_policing'  # Replace with your directory path
files = list_all_files(directory_path)

dfs = []
for f in reversed(files):

    df_init = pd.read_feather(f)
    #df_init.to_feather(f[:-4] + '.feather')
    df_init['file'] = f
    #df = dd.concat([df, df_init], axis=1, ignore_index=True)
    dfs.append(df_init)
    print(f)

df = pd.concat(dfs, axis=1, ignore_index=True)

df.to_feather('policing_data_products/agg_nc_unmodified.feather')
