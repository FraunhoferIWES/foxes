import numpy as np

def wd2wdvec(wd, ws=1., axis=-1):
    
    wdr  = wd * np.pi / 180.
    n    = np.stack([np.sin(wdr), np.cos(wdr)], axis=axis)
    
    if np.isscalar(ws):
        return ws * n

    return np.expand_dims(ws, axis) * n

def wd2uv(wd, ws=1., axis=-1):
    return -wd2wdvec(wd, ws, axis)

def uv2wd(uv, axis=-1):

    if axis == -1:
        u = uv[..., 0]
        v = uv[..., 1]
    else:
        s = tuple(0 if a == axis else slice(None) for a in range(len(uv.shape)))
        u = uv[s]
        s = tuple(1 if a == axis else slice(None) for a in range(len(uv.shape)))
        v = uv[s]

    return np.mod(180 + np.rad2deg(np.arctan2(u, v)), 360)

def wdvec2wd(wdvec, axis=-1):
    return uv2wd(-wdvec, axis)

