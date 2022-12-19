from math import sin, cos
import cv2 as cv
import mahotas
import mahotas.demos
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.patches as patches
import pandas as pd
from scipy import signal
import pickle
from scipy.linalg import norm
import numpy as np
from math import atan
import imutils
import sys
import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        uic.loadUi(
            r'Final.ui', self)

        self.openFile.clicked.connect(lambda: self.read_data())
        
        # Global variables
        self.temp_size = 35
        self.window_size = 50
        self.land = 12
        self.search_window = []
        self.position = []
        
        self.graph = pg.PlotItem()
        self.graph.hideAxis('left')
        self.graph.hideAxis('bottom')

        # Intiating canvas
        self.originalCanvas = MplCanvas(
                                        self.imageView,  width=5.5, height=4.5, dpi=90)
        self.originalLayout = QtWidgets.QVBoxLayout()
        self.originalLayout.addWidget(self.originalCanvas)
        self.originalCanvas.draw()

    def read_data(self):
        data = pd.read_csv(
            r"F:\materials\Fourth year\Biometrics\code\train_senior.csv")

        self.test_data = pd.read_csv(
            r"F:\materials\Fourth year\Biometrics\Code\test1_senior.csv")
        
        self.names = np.array(data.iloc[:,0])

        self.shapes = np.array(data.iloc[:, 1:])

        imagePath = QFileDialog.getOpenFileName(
            self, "Open File", "This PC",
            "All Files (*);;PNG Files(*.png);; Jpg Files(*.jpg)")
        
        self.test_img = cv.imread(imagePath[0], cv.IMREAD_GRAYSCALE)
        
        self.moments = pd.read_csv("train_moments.csv")
        self.moments = np.asarray(self.moments)
        
        with open('images_templates.pkl', 'rb') as f:
            self.images_templates = pickle.load(f)
        with open('landmark_moments.pkl', 'rb') as f:
            self.images_templates_moments = pickle.load(f)
        
        self.get_similar_shapes()
        self.template_matching()
        self.display()


    def get_translation(self,shape):
        '''
        Calculates a translation for x and y
        axis that centers shape around the
        origin
        Args:
            shape(2n x 1 NumPy array) an array 
            containing x coodrinates of shape
            points as first column and y coords
            as second column
        Returns:
            translation([x,y]) a NumPy array with
            x and y translationcoordinates
        '''

        mean_x = np.mean(shape[::2]).astype(int)
        mean_y = np.mean(shape[1::2]).astype(int)

        return np.array([mean_x, mean_y])


    def translate(self,shape):
        '''
        Translates shape to the origin
        Args:
            shape(2n x 1 NumPy array) an array 
            containing x coodrinates of shape
            points as first column and y coords
            as second column
        '''
        mean_x, mean_y = self.get_translation(shape)
        shape[::2] -= mean_x
        shape[1::2] -= mean_y

    def get_rotation_scale(self,reference_shape, shape):
        '''
        Calculates rotation and scale
        that would optimally align shape
        with reference shape
        Args:
            reference_shape(2nx1 NumPy array), a shape that
            serves as reference for scaling and 
            alignment
            
            shape(2nx1 NumPy array), a shape that is scaled
            and aligned
            
        Returns:
            scale(float), a scaling factor
            theta(float), a rotation angle in radians
        '''

        a = np.dot(shape, reference_shape) / norm(reference_shape)**2

        #separate x and y for the sake of convenience
        ref_x = reference_shape[::2]
        ref_y = reference_shape[1::2]

        x = shape[::2]
        y = shape[1::2]

        b = np.sum(x*ref_y - ref_x*y) / norm(reference_shape)**2

        scale = np.sqrt(a**2+b**2)
        theta = atan(b / max(a, 10**-10))  # avoid dividing by 0

        return round(scale, 1), round(theta, 2)


    def get_rotation_matrix(self,theta):

        return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])


    def scale(self,shape, scale):

        return shape / scale


    def rotate(self, shape, theta):
        '''
        Rotates a shape by angle theta
        Assumes a shape is centered around 
        origin
        Args:
            shape(2nx1 NumPy array) an shape to be rotated
            theta(float) angle in radians
        Returns:
            rotated_shape(2nx1 NumPy array) a rotated shape
        '''

        matr = self.get_rotation_matrix(theta)

        #reshape so that dot product is eascily computed
        temp_shape = shape.reshape((-1, 2)).T

        #rotate
        rotated_shape = np.dot(matr, temp_shape)

        return rotated_shape.T.reshape(-1)


    def procrustes_analysis(self, reference_shape, shape):
        '''
        Scales, and rotates a shape optimally to
        be aligned with a reference shape
        Args:
            reference_shape(2nx1 NumPy array), a shape that
            serves as reference alignment
            
            shape(2nx1 NumPy array), a shape that is aligned
            
        Returns:
            aligned_shape(2nx1 NumPy array), an aligned shape
            translated to the location of reference shape
        '''
        #copy both shapes in caseoriginals are needed later
        temp_ref = np.copy(reference_shape)
        temp_sh = np.copy(shape)

        
        self.translate(temp_ref)
        self.translate(temp_sh)

        #get scale and rotation
        scale, theta = self.get_rotation_scale(temp_ref, temp_sh)

        #scale, rotate both shapes
        temp_sh = temp_sh / scale
        aligned_shape = self.rotate(temp_sh, theta)

        return aligned_shape


    def procrustes_distance(self,reference_shape, shape):

        ref_x = reference_shape[::2]
        ref_y = reference_shape[1::2]

        x = shape[::2]
        y = shape[1::2]

        dist = np.sum(np.sqrt((ref_x - x)**2 + (ref_y - y)**2))

        return dist


    def generalized_procrustes_analysis(self,shapes):
        '''
        Performs superimposition on a set of 
        shapes, calculates a mean shape
        Args:
            shapes(a list of 2nx1 Numpy arrays), shapes to
            be aligned
        Returns:
            mean(2nx1 NumPy array), a new mean shape
            aligned_shapes(a list of 2nx1 Numpy arrays), super-
            imposed shapes
        '''
        #initialize Procrustes distance
        current_distance = 0

        #initialize a mean shape
        mean_shape = shapes[0]

        num_shapes = len(shapes)

        #create array for new shapes, add
        new_shapes = np.zeros(np.array(shapes).shape)

        while True:

            #add the mean shape as first element of array
            new_shapes[0] = mean_shape

            #superimpose all shapes to current mean
            for sh in range(1, num_shapes):
                mean_x = np.mean(shapes[sh][::2]).astype(int)
                mean_y = np.mean(shapes[sh][1::2]).astype(int)

                new_sh = self.procrustes_analysis(mean_shape, shapes[sh])
                
                new_sh[::2] += mean_x
                new_sh[1::2] += mean_y
                new_shapes[sh] = new_sh

            #calculate new mean
            new_mean = np.mean(new_shapes, axis=0)

            new_distance = self.procrustes_distance(new_mean, mean_shape)

            #if the distance did not change, break the cycle
            if new_distance == current_distance:
                break

            #align the new_mean to old mean
            mean_x = np.mean(mean_shape[::2]).astype(int)
            mean_y = np.mean(mean_shape[1::2]).astype(int)
            new_mean = self.procrustes_analysis(mean_shape, new_mean)
            new_mean[::2] += mean_x
            new_mean[1::2] += mean_y
            #update mean and distance
            mean_shape = new_mean
            current_distance = new_distance

        return mean_shape, new_shapes


    def get_similar_shapes(self):
        # decide which images from train is close to you
        test_moment = np.array(mahotas.features.zernike_moments(self.test_img, 10))

        dists = []
        for i in range(len(self.moments)):
            dist = np.linalg.norm(self.moments[i] - test_moment)
            dists.append(dist)
        dists = np.asarray(dists)

        indx = np.argsort(dists)
        similar_idx = indx[0:8]
        # similar_idx contain 8 most similar images to our test image
        self.names = self.names[similar_idx]
        similar_shapes = self.shapes[similar_idx]

        self.mean_similar_shape, self.new_similar_shapes = self.generalized_procrustes_analysis(
            similar_shapes)
        
           
    def get_center_points(self):
        self.center_points = []
        # get x and y coordinate of each landmark in image[idx]
        shape = self.mean_similar_shape
        x_pos = shape[::2]
        y_pos = shape[1::2]
        # calculate the centers of templates for each landmark
        c1 = [x_pos-int(0.5*self.window_size), y_pos-int(0.5*self.window_size)]
        c2 = [x_pos-int(0.5*self.window_size), y_pos+int(0.5*self.window_size)]
        c3 = [x_pos+int(0.5*self.window_size), y_pos-int(0.5*self.window_size)]
        c4 = [x_pos+int(0.5*self.window_size), y_pos+int(0.5*self.window_size)]
        self.center_points = np.concatenate((c1, c2, c3, c4))
        self.center_points = np.asarray(self.center_points)
        
        
    def template_matching(self):
        shapes = np.zeros(shape=(2400,1935))
        shape = np.reshape(self.mean_similar_shape, (-1, 2)).astype(int)
        for i in self.names:
            image = cv.imread(str('../cepha400/cepha400/'+i), cv.IMREAD_GRAYSCALE)
            shapes += cv.equalizeHist(image)    
        avg_shape = shapes/ 8.0
        
        eq_test_img = cv.equalizeHist(self.test_img)
        for i in range(19):
            temp_land = avg_shape[shape[i][1]-self.temp_size: shape[i][1]+self.temp_size, shape[i][0]-self.temp_size:shape[i][0]+self.temp_size]
            
            window = eq_test_img[shape[i][1]-self.window_size:shape[i][1]+self.window_size, shape[i][0] -
                                self.window_size:shape[i][0]+self.window_size]
            
            res = cv.matchTemplate(
                window, temp_land.astype(np.uint8), cv.TM_CCORR_NORMED)
            _, _, _, max_loc = cv.minMaxLoc(res)

            pos = [max_loc[0]+shape[i][0]-self.window_size+self.temp_size,
                   max_loc[1]+shape[i][1]-self.window_size+self.temp_size]
            self.position.append(pos)
            rect = patches.Rectangle(
                (shape[i][0]-self.window_size, shape[i][1]-self.window_size), self.window_size*2, self.window_size*2, linewidth=1, edgecolor='r', facecolor='none')
            self.search_window.append(rect)
            
    def display(self):
        for i in range(19):
            self.originalCanvas.axes.add_patch(self.search_window[i])
            self.originalCanvas.axes.scatter(self.position[i][0],
                                         self.position[i][1], marker='*', color="red", s=30)
        self.originalCanvas.axes.imshow(self.test_img, cmap="gray")

        # test_shape = np.array(self.test_data.iloc[1, 1:])
        # test_shape = np.reshape(test_shape, (-1, 2))
        # self.originalCanvas.axes.scatter(test_shape[:, 0],
        #                                  test_shape[:, 1], marker="*", color="green", s=30)

        mean_similar_shape = np.reshape(self.mean_similar_shape, (-1, 2))
        self.originalCanvas.axes.scatter(mean_similar_shape[:, 0],
                                         mean_similar_shape[:, 1], marker='*', color='blue', s=30)
        self.originalCanvas.draw()

        self.imageView.setCentralItem(self.graph)
        self.imageView.setLayout(self.originalLayout)

    
   
        
def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
