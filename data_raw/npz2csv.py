# convert npz file into csv for R
import glob
import re
import numpy as np
import pandas as pd
import argparse

def npz2csv (directory):
    all_npz = glob.glob(directory+'/*.npz')
    for one_npz in all_npz:
        mat = pd.DataFrame(np.load (one_npz)['arr_0'])
        one_csv = re.sub ('.npz', '.csv', one_npz)
        mat.to_csv (one_csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    args = parser.parse_args()
    npz2csv (args.dir)
