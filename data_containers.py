import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import cv2

import drift_profile

import matplotlib
matplotlib.use("Agg")

def plot_to_numpy(xrange, datas, width=800, height=600, dpi=100):
    """
    Convert numerical data plot to numpy array

    Parameters:
    data: numpy array or list of numerical values
    width: plot width in pixels
    height: plot height in pixels
    dpi: dots per inch
    
    Returns:
    numpy array of shape (height, width, 4) containing RGBA values
    """
    # Create figure
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    
    # Plot the data
    args = []
    for data in datas:
        args.append(xrange)
        args.append(data)
    plt.plot(*args, linewidth=2)
    plt.grid(True)

    # Save plot to bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='rgba', dpi=dpi)
    buf.seek(0)
    
    # Convert buffer to numpy array
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img_array = img_array.reshape(height, width, 4)
    img_array = img_array[:,:,0:3]
    
    # Clean up
    plt.close(fig)
    buf.close()
    
    return img_array

class DriftTrack:
    half_w : int = 0
    slices : np.ndarray = None

    def __init__(self, gray : np.ndarray, margin : int = 0, points : np.ndarray = None, normals : np.ndarray = None, transposed : bool = False):
        self.gray = gray
        self.points = points
        self.transposed = transposed
        self.normals = normals
        self.margin = margin

    def draw(self, color, transparency):
        rgb = cv2.cvtColor(self.gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return self.draw_in_place(rgb, 0, 0, color, transparency)

    def draw_in_place(self, rgb, left, top, color, transparency):
        color = np.array(color)
        if self.points is not None:
            for y, x in self.points:
                xx = int(x+self.margin+left)
                yy = int(y+self.margin+top)
                if xx < 0 or yy < 0 or xx >= rgb.shape[1] or yy >= rgb.shape[0]:
                    continue
                rgb[yy, xx] = rgb[yy, xx] * transparency + color * (1-transparency)

        # draw normals
        if self.normals is not None and self.points is not None:
            for index, ((y,x), (ny,nx)) in enumerate(zip(self.points, self.normals)):
                if index % 10 != 0:
                    continue
                x1 = int(x - nx*self.half_w + self.margin)
                y1 = int(y - ny*self.half_w + self.margin)
                x2 = int(x + nx*self.half_w + self.margin)
                y2 = int(y + ny*self.half_w + self.margin)

                cv2.line(rgb, (x1+left,y1+top), (x2+left,y2+top), (0,200,0), 1)
        return rgb

    def plot_slice(self, w : int, h : int, layer : int = -1):
        xr = range(2*self.half_w+1)
        if layer == -1:
            values = np.mean(self.slices, axis=0)
            top = np.amax(self.slices, axis=0)
            low = np.amin(self.slices, axis=0)
            rgb = plot_to_numpy(xr, [values, top, low], w, h)
        else:
            values = self.slices[layer]
            rgb = plot_to_numpy(xr, [values], w, h)
        return rgb

class DriftProfile:
    def __init__(self, profile : np.ndarray, error : np.ndarray):
        self.profile = profile
        self.error = error

    def plot_profile(self, w : int, h : int, smooth_err : int = 0):
        L = self.profile.shape[0]
        xr = range(L)
        if smooth_err == 0:
            rgb = plot_to_numpy(xr, [self.profile, self.profile + self.error, self.profile - self.error], w, h)
        else:
            top = drift_profile.smooth_track_profile(self.profile + self.error, smooth_err)
            bottom = drift_profile.smooth_track_profile(self.profile - self.error, smooth_err)
            rgb = plot_to_numpy(xr, [self.profile, top, bottom], w, h)
        return rgb
