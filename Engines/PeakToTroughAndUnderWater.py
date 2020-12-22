
import numpy as np
import pandas as pd

def PeakToTrough(r):
    r = np.array(r) + 1
    ser = pd.Series(r).cumprod()
    max2here = ser.cummax()
    drop_down_ser = (ser - max2here)/max2here
    max_dropdown = drop_down_ser.min()
    return max_dropdown


def UnderWater(r):
    r = r.values
    df = pd.DataFrame(r, columns=('r',))
    df["cumret"] = (df["r"]+1).cumprod()
    df["cummax"] = df["cumret"].cummax()
    df["underwater"] = df["cumret"] < df["cummax"]

    last_ix = 0; clock = 0
    for ix, val in enumerate(df["underwater"].to_numpy()):
        if (not val) or (ix == len(df["underwater"])-1):
            clock = max(clock, ix-last_ix)
            last_ix = ix
    return clock / len(r)



