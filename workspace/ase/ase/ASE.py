#===============================================================================
# Copyright (C) 2012 Kale J. Franz
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#===============================================================================

from __future__ import division
import sys
from PyQt4.QtCore import QString, SIGNAL
from PyQt4.QtGui import QHBoxLayout, QApplication, QCheckBox, QFileDialog, QLabel, QLineEdit, QMainWindow, QPushButton, QVBoxLayout, QWidget
from numpy import arange, bitwise_and, column_stack, convolve, genfromtxt, linspace, log, nan_to_num, nonzero, ones, r_, roll, savetxt, sqrt, vstack, hanning
from scipy import interpolate

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

#===============================================================================
# Version
#===============================================================================
hpVersion = 120207

#===============================================================================
# Global Constants
#===============================================================================
pi = 3.14159265
e0 = 1.60217653e-19;  #electron charge
eps0 = 8.85e-12;
m0 = 9.10938188e-31;   #free electron mass (kg)
hbar = 6.6260693e-34/(2*pi); #Planck's constant (J s)
kb = 1.386505e-23 / e0; #eV/K
wn2energy = 0.1/1.24


def get_peaks_valleys(intensities):
    tck = interpolate.splrep(intensities[:,0],intensities[:,1],s=0)
    wn = linspace(intensities[0][0],intensities[-1][0],5e6)
    inten = interpolate.splev(wn,tck,der=0)
    
    intenCWBool = roll(inten,1) > inten
    intenCCWBool = roll(inten,-1) > inten
    minIntenBool = bitwise_and(intenCWBool, intenCCWBool)
    minIntenIdxs = nonzero(minIntenBool == True)[0]
    minInten = vstack([wn[minIntenIdxs], inten[minIntenIdxs]]).T
    
    intenCWBool = roll(inten,1) < inten
    intenCCWBool = roll(inten,-1) < inten
    maxIntenBool = bitwise_and(intenCWBool, intenCCWBool)
    maxIntenIdxs = nonzero(maxIntenBool == True)[0]
    maxInten = vstack([wn[maxIntenIdxs], inten[maxIntenIdxs]]).T
    
    if minInten[0,1] < maxInten[0,1]:
        #first point is a minimum
        #so take out first Ymin
        minInten = minInten[1:,:]
    else:
        #first point is a maximum
        #so take out last Ymin
        minInten = minInten[0:-1,:]
    
    return (minInten, maxInten)
    
def get_gain(minIntens, maxIntens, length):
    mins = minIntens[:,1]
    maxs = maxIntens[:,1]
    wavenum = minIntens[:,0]
    V = (maxs[0:-1] + maxs[1:]) / (2 * mins)
    
    #force all values of V to be positive
    V[nonzero(V < 0)[0]] = 0
    
    gamma = -1/length * log( (sqrt(V)+1) / (sqrt(V)-1) )
    gamma = gamma.real
    gamma = nan_to_num(gamma)
    
    return vstack([wavenum, gamma]).T
    
def smooth_gain(gain):
    wn = gain[:,0]
    gamma = gain[:,1]
    
    tck = interpolate.splrep(wn,gamma,s=0)
    wavenum = arange(wn[0],wn[-1],1)
    gamma2 = interpolate.splev(wavenum,tck,der=0)
    
    gamma2 = smooth(gamma2, window='flat')
    gamma2 = gamma2[5:-5]
    gamma2 = smooth(gamma2, window='hanning')
    gamma2 = gamma2[5:-5]
    
    return vstack([wavenum, gamma2]).T
    
    
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=ones(window_len,'d')
    else:
        w=eval(''+window+'(window_len)')

    y=convolve(w/w.sum(),s,mode='valid')
    return y
    

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        
        self.cavityLength = 0.1
        self.fileName = None
        
        self.create_main_frame()
        self.setWindowTitle('ASE')
        

        
    def create_main_frame(self):
        
        #hBoxFor2
        hBox2 = QHBoxLayout()
        hBox2.addWidget(QLabel('Cavity Length (cm)'))
        self.cavityLengthBox = QLineEdit(str(self.cavityLength))
        self.connect(self.cavityLengthBox, SIGNAL("editingFinished()"), self.set_cavityLength)
        hBox2.addWidget(self.cavityLengthBox)
        hBox2.addStretch()        
        
        #vBox2
        selectFileButton = QPushButton('Select File')
        self.connect(selectFileButton, SIGNAL("clicked(bool)"), self.select_file)
        hBox2.addWidget(selectFileButton)
        
        self.spectrumWidget = QWidget()
        dpi = 100
        self.spectrumFig = Figure((5.0, 4.0), dpi=dpi)
        self.spectrumCanvas = FigureCanvas(self.spectrumFig)
        self.spectrumCanvas.setParent(self.spectrumWidget)
        self.spectrumAxes = self.spectrumFig.add_subplot(111)
        self.spectrumToolbar = NavigationToolbar(self.spectrumCanvas, self.spectrumWidget)
        
        vBox2 = QVBoxLayout()
        vBox2.addLayout(hBox2)
        vBox2.addWidget(self.spectrumCanvas)
        vBox2.addWidget(self.spectrumToolbar)
        
        
        
        #hBoxFor3
        hBox3 = QHBoxLayout()

        self.smoothResultsBox = QCheckBox('Smooth Results')
        hBox3.addWidget(self.smoothResultsBox)
        hBox3.addStretch()
        
        plotGainButton = QPushButton('Plot Gain')
        self.connect(plotGainButton, SIGNAL('clicked()'), self.plot_gain)
        hBox3.addWidget(plotGainButton)
        hBox3.addStretch()
        
        saveFileButton = QPushButton('Save Gain')
        self.connect(saveFileButton, SIGNAL("clicked(bool)"), self.save_file)
        hBox3.addWidget(saveFileButton)
        
        
        #vBox3
        self.gainWidget = QWidget()
        self.gainFig = Figure((5.0, 4.0), dpi=dpi)
        self.gainCanvas = FigureCanvas(self.gainFig)
        self.gainCanvas.setParent(self.gainWidget)
        self.gainAxes = self.gainFig.add_subplot(111)
        self.gainToolbar = NavigationToolbar(self.gainCanvas, self.gainWidget)
        
        vBox3 = QVBoxLayout()
        vBox3.addLayout(hBox3)
        vBox3.addWidget(self.gainCanvas)
        vBox3.addWidget(self.gainToolbar)
        
        
        mainLayout = QHBoxLayout()
        mainLayout.addLayout(vBox2)
        mainLayout.addLayout(vBox3)
        
        mainWidget = QWidget()
        mainWidget.setLayout(mainLayout)
        self.setCentralWidget(mainWidget)
        
    def set_cavityLength(self):
        self.cavityLength = float(self.cavityLengthBox.text())

    def select_file(self):
        self.fileName =unicode(QFileDialog.getOpenFileName(self,"ASE -- Choose file", '.',
                                                    "CSV file (*.csv)\nAll files (*.*)"))

        self.intensities = genfromtxt(self.fileName, delimiter=',')
        
        #assure all values are > 0
        self.intensities[:,1] -= self.intensities[:,1].min()
        self.minIntensities, self.maxIntensities = get_peaks_valleys(self.intensities)
       
        self.plot_spectrum()
        
        
    def save_file(self):
        fname = unicode(QFileDialog.getSaveFileName(self,"ASE -- Save Gain File", 
                        QString(self.fileName), "CSV file (*.csv)\nAll files (*.*)"))
        savetxt(fname, column_stack((self.conversion * self.gain[:,0], self.gain[:,1])), delimiter=',')
        
    
    def plot_spectrum(self):
        self.spectrumAxes.clear()
        self.spectrumAxes.plot(self.intensities[:,0], self.intensities[:,1])
        self.spectrumAxes.plot(self.minIntensities[:,0],self.minIntensities[:,1],'g.')
        self.spectrumAxes.plot(self.maxIntensities[:,0],self.maxIntensities[:,1],'r.')
        self.spectrumAxes.set_xlabel('Wavenumber (cm^-1)')
        self.spectrumAxes.set_ylabel('Intensity (a.u.)')
        #self.neff = column_stack((self.minIntensities[:,0], 1/(2*self.cavityLength*diff(self.maxIntensities[:,0]))))
        #self.neff = smooth_gain(self.neff)
        #self.spectrumAxes.plot(self.neff[:,0], self.neff[:,1])
        self.spectrumCanvas.draw()
        
    def plot_gain(self):
        self.gain = get_gain(self.minIntensities, self.maxIntensities, self.cavityLength)
        if self.smoothResultsBox.isChecked():
            self.gain = smooth_gain(self.gain)
            
        self.conversion = wn2energy
        
        self.gainAxes.clear()
        self.gainAxes.plot(self.gain[:,0]*self.conversion, self.gain[:,1])
        if self.conversion == wn2energy:
            self.gainAxes.set_xlabel('Energy (meV)')
            self.gainAxes.set_ylabel('Gain (cm^-1)')
        self.gainCanvas.draw()
        
        
        
def main():
    app = QApplication(sys.argv)
    form = MainWindow()
    form.show()
    app.exec_()

main()