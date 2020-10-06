import matplotlib.pyplot as plt
import logging
import numpy as np

log = logging.getLogger(__name__)



def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

def plot2DData(data,color, path, name):

    f = plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=color[:], cmap="inferno")
    plt.xticks([], [])
    plt.yticks([], [])
    f.savefig(path+name, bbox_inches='tight')
    plt.close(f)

    log.info("plotted file %s" % name)


def plot2DLatentSpace(x,y,color, path, name):


    f = plt.figure()


    plt.scatter(x, y, c=color[:], cmap='inferno')

    # plt.annotate('', xy=(-2, 0), xycoords='data',
    #              xytext=(-3, 0), textcoords='data',
    #              arrowprops=dict(facecolor='black', width=3))
    plt.xticks([], [])
    plt.yticks([], [])
    f.savefig(path+name, bbox_inches='tight')
    plt.close(f)

    log.info("plotted file %s" % name)
