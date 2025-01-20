import numpy as np
import pandas as pd
from gzip import GzipFile
from nibabel import FileHolder, Nifti1Image

def read_file(byte_file):
    # See https://stackoverflow.com/questions/62579425/simpleitk-read-io-byteio
    fh = FileHolder(fileobj=GzipFile(fileobj=byte_file))
    img = Nifti1Image.from_file_map({'header': fh, 'image': fh})
    arr = np.array(img.dataobj)
    return arr, img.header


def read_from_eclipse(file_name):
    df = pd.DataFrame()
    f = open(file_name, "r")
    for line in f:
        if "Structure:" in line:
            name = line.split(" ")[-1]
            for line in f:
                if "Relative dose [%]" in line:
                    row_cnt = 0
                    for line in f:
                        if len(line.split()) > 2:
                            df.loc[row_cnt, name + "_dose"] = (
                                float(line.split()[1]) / 100.0
                            )
                            df.loc[row_cnt, name + "_vol"] = float(line.split()[2])
                            row_cnt += 1
                        else:
                            break
                    break
    f.close()
    return df
