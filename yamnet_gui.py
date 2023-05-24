import sys
import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt

import params as yamnet_params
import yamnet as yamnet_model
import tensorflow as tf

from PyQt5.QtCore import Qt 
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QGridLayout, QWidget, QAction, QHBoxLayout, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector


import warnings
warnings.filterwarnings('ignore')

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.t_ini = 0.0 # Tiempo inicial de la ventana
        self.m_ini = 0   # Número de la muestra inicial de la ventana
        self.t_ven = 30.  # Tiempo de datos de una ventana
        self.m_ven = 0   # Número de muestras de una ventana
        self.t_fin = 0.0 # Tiempo final de la ventana
        self.m_fin = 0   # Número de la muestra final de la ventana
        self.t_tot = 0.0 # Tiempo total del fichero
        self.m_tot = 0   # Número total de muestras del fichero

        self.setWindowTitle('YAMNet GUI')
        self.setGeometry(50, 50, 1000, 1000)
         
        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        self.setCentralWidget(self.canvas)
        
        self.next_frame_button = QPushButton('Next frame', self)
        self.next_frame_button.clicked.connect(self.next_frame)
         
        self.prev_frame_button = QPushButton('Prev frame', self)
        self.prev_frame_button.clicked.connect(self.previous_frame)

        self.save_audio_button = QPushButton('Save Audio', self)
        self.save_audio_button.clicked.connect(self.save_audio)
         
        self.play_audio_button = QPushButton('Play Audio', self)
        self.play_audio_button.clicked.connect(self.play_audio)
         
        
        grid_layout = QGridLayout()
        grid_layout.addWidget(self.canvas, 0, 0, 1, 2)
        grid_layout.addWidget(self.nav_toolbar, 3, 0, 1, 2)
        grid_layout.addWidget(self.next_frame_button, 1, 1, 1, 1)  
        grid_layout.addWidget(self.prev_frame_button, 1, 0, 1, 1) 
        grid_layout.addWidget(self.save_audio_button, 2, 1, 1, 1)  
        grid_layout.addWidget(self.play_audio_button, 2, 0, 1, 1)  

        central_widget = QWidget()
        central_widget.setLayout(grid_layout)
        self.setCentralWidget(central_widget)

        self.statusBar().showMessage('Open a WAV file to analyze')
         
        self.create_menu()

        self.fig = self.canvas.figure
        self.ax1 = self.fig.add_subplot(3,1,1)
        self.ax2 = self.fig.add_subplot(3,1,2)
        self.ax3 = self.fig.add_subplot(3,1,3)

        #self.fig, ax = plt.subplots(3,1, sharex=True)
        #self.ax1 = ax[0]
        #self.ax2 = ax[1]
        #self.ax3 = ax[2]
        #self.canvas.figure = self.fig
         # Set up the YAMNet model.

        self.ax1.get_shared_x_axes().join(self.ax1, self.ax2, self.ax3)

        
    def create_menu(self):
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('File')
        openFileAction = QAction('Open', self)
        openFileAction.triggered.connect(self.open_file)
        fileMenu.addAction(openFileAction)
        closeFileAction = QAction('Close', self)
        closeFileAction.triggered.connect(self.close_file)
        fileMenu.addAction(closeFileAction)
        exitFileAction = QAction('Exit', self)
        exitFileAction.triggered.connect(self.exit)
        fileMenu.addAction(exitFileAction)
    
    def open_file(self):
        openFile, _ = QFileDialog.getOpenFileName(self, 'Open WAV file', '', 'WAV Files (*.wav)')
        if openFile:
            self.wav_file_name = openFile
            self.statusBar().showMessage('Opening WAV file...')
            audio_info = sf.info(self.wav_file_name)
            self.sr = audio_info.samplerate
            self.m_ven = int(self.t_ven*self.sr)
            self.t_ven = self.m_ven/self.sr
            self.m_ini = 0
            self.t_ini = 0.0
            self.m_fin = self.m_ven
            self.t_fin = self.t_ven
            self.t_tot = audio_info.duration
            self.m_tot = audio_info.frames
            self.crt_start = 0
            self.crt_end = self.m_fin
            self.statusBar().showMessage('Loading model file...')
            self.carga_modelo()
            self.read_wav()
            
    def read_wav(self):
        # The graph is designed for a sampling rate of 16 kHz, but higher rates should work too.
        # We also generate scores at a 10 Hz frame rate.
        
        self.waveform, _ = sf.read(self.wav_file_name, 
                                   frames=self.m_ven, 
                                   start=self.m_ini)
        
        self.statusBar().showMessage('WAV file read: ' + self.wav_file_name)

        self.analyze_wav()

    def carga_modelo(self):
        self.params = yamnet_params.Params(sample_rate=self.sr, patch_hop_seconds=0.1)
        self.class_names = yamnet_model.class_names('yamnet_class_map.csv')
        self.yamnet = yamnet_model.yamnet_frames_model(self.params)
        self.yamnet.load_weights('yamnet.h5')
        
    def analyze_wav(self):

        tiempo = self.t_ini+np.arange(self.m_ven)/self.sr
         
        scores, embeddings, spectrogram = self.yamnet(self.waveform)
        scores = scores.numpy()
        spectrogram = spectrogram.numpy()
         
        self.ax1.clear()
        self.ax1.plot(tiempo, self.waveform)
        self.ax1.set_ylabel('Amplitude')
         
        self.ax2.clear()
        self.ax2.imshow(spectrogram.T, 
                        aspect='auto', 
                        interpolation='nearest', 
                        origin='lower', 
                        extent=[self.t_ini,self.t_fin+0.1,0,self.sr/2])
         
        mean_scores = np.mean(scores, axis=0)
        top_N = 10
        top_class_indices = np.argsort(mean_scores)[::-1][:top_N]

        self.ax3.clear()
        self.ax3.imshow(scores[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r', extent=[self.t_ini,self.t_fin,-0.5,top_N-0.5])
        # Compensate for the patch_window_seconds (0.96s) context window to align with spectrogram.
        yticks = range(top_N)
        self.ax3.set_yticks(yticks[::-1])
        self.ax3.set_yticklabels(labels=[self.class_names[top_class_indices[x]] for x in yticks])

        self.canvas.draw()
        self.statusBar().showMessage('Analysis done') 
         
        self.span_selector = SpanSelector(self.ax1, 
                                          self.crop_audio, 
                                          'horizontal', 
                                          useblit=True,
                                          props=dict(alpha=0.5, facecolor="tab:red"),
                                          interactive=True,
                                          drag_from_anywhere=True)  

    def next_frame(self):
        if self.m_ini+self.m_ven < self.m_tot:
            self.m_ini += self.m_ven
            self.t_ini = self.m_ini/self.sr
            self.t_fin = (self.m_ini+self.m_ven)/self.sr
            self.read_wav()

    def previous_frame(self):
        if self.m_ini-self.m_ven > 0:
            self.m_ini -= self.m_ven
            self.t_ini = self.m_ini/self.sr
            self.t_fin = (self.m_ini+self.m_ven)/self.sr
            self.read_wav()

    def crop_audio(self,t_min, t_max):
        self.crt_start = int((t_min-self.t_ini) * self.sr)
        self.crt_end = int((t_max-self.t_ini) * self.sr)

    def play_audio(self):
        data = self.waveform[self.crt_start:self.crt_end]
        data /= np.max(np.abs(data))
        sd.play(data, self.sr,blocksize=self.sr//10)

    def save_audio(self):
        filedialog = QFileDialog()
        filesave, _ = filedialog.getSaveFileName(self, 'Save File', '', 'WAV files (*.wav)')
        if filesave:
            sf.write(filesave, self.waveform[self.crt_start:self.crt_end], self.sr)
            self.statusBar().showMessage("Audio cropped successfully. Saved as: " + filesave)
     
    def close_file(self):
        return
    
    def exit(self):
        sys.exit()
        return
    
    #def show history of saves in the rigth
    def show_history(self):
        
        return
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
