import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_events import MplEventDispatcher, mpl

from lib.polyfit2d import polyfit2d
from lib.zscale import zscale


def getGoodOrderedRegion(region):
    x1 = region[0][0]
    x2 = region[1][0]
    y1 = region[0][1]
    y2 = region[1][1]
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return [(x1, y1), (x2, y2)]


class SelectRegionsDispatcher(MplEventDispatcher):
    def __init__(self, figure, sample):
        super().__init__(figure)
        self.fig = figure
        self.region = []
        self.regionsList = []
        self.sample = sample

    def on_mouse_button_release(self, event: mpl.MouseEvent):
        if len(self.region) < 2:
            self.region.append((int(event.xdata), int(event.ydata)))
        if len(self.region) == 2:
            self.region = getGoodOrderedRegion(self.region)
            rect = patches.Rectangle(self.region[0], self.region[1][0] - self.region[0][0],
                                     self.region[1][1] - self.region[0][1], linewidth=0.5,
                                     edgecolor='r', facecolor='none')
            self.fig.axes[0].add_patch(rect)
            self.fig.canvas.draw_idle()
            self.regionsList.append(self.region)
            self.region = []
            if self.sample == len(self.regionsList):
                plt.close('all')


def selectRegionsAndInterpolate(data, sample, k, v=None):
    yLen, xLen = data.shape
    fig, ax = plt.subplots()
    disp = SelectRegionsDispatcher(fig, sample)
    if not v:
        vmin, vmax = zscale(data)
    else:
        vmin, vmax = v
    im = ax.imshow(data, cmap=plt.cm.bone, origin='lower',
                   norm=LogNorm(vmin=vmin, vmax=vmax), )
    ax.set_title("Select Regions And Interpolate")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('value', rotation=-90, va="bottom")
    plt.show()
    kx, ky = k
    res = polyfit2d(disp.regionsList, data, kx=kx, ky=ky)[0]
    return res.reshape((kx + 1, ky + 1))


def getSkyBackground(data, sample=10, k=(1, 1)):
    poly = selectRegionsAndInterpolate(data, sample=sample, k=k)
    yLen, xLen = data.shape
    x = range(0, xLen)
    y = range(0, yLen)
    x, y = np.meshgrid(x, y)
    sky = np.zeros_like(x)
    for index, (j, i) in enumerate(np.ndindex(poly.shape)):
        sky = sky + poly[i, j] * x ** i * y ** j
    # alternative way:
    # fitted_surf = np.polynomial.polynomial.polyval2d(x, y, poly)
    return sky


def saveSky(skyBack, name="skyBack"):
    skyBack.tofile(name + '.bin')
    with open(name + '.shape', 'w') as shapeFile:
        shapeFile.write(str(skyBack.shape))


def loadSky(name="skyBack"):
    with open(name + '.shape', 'r') as shapeFile:
        shape = tuple(map(int, shapeFile.read()[1:-1].split(',')))
    skyBack = np.fromfile(file=name + '.bin', dtype=np.float64).reshape(shape)
    return skyBack
