import sys
import traceback
import os
from PyQt5.QtWidgets import (QWidget, QFileDialog, QPushButton, QTextEdit,
                             QGridLayout, QApplication, QLabel, QComboBox)
# files for this package
import loss_analysis

class Loss_analysis_gui(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.solar_cell = loss_analysis.loss_analysis_handeller()

        path = os.sep.join(os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-1])
        example_dir = os.path.join(path, 'example_cell')
        print(path,example_dir)
        grid = QGridLayout()
        # grid.setSpacing(10)

        # reflectance
        self.btn_refl = QPushButton("Load reflectance")
        self.btn_refl.clicked.connect(self.get_refl)
        grid.addWidget(self.btn_refl, 1, 0)
        self.label_refl = QLabel('example_reflectance.csv', self)
        self.prev_fullpath_refl = os.path.join(example_dir, 'example_reflectance.csv')
        grid.addWidget(self.label_refl, 1, 1)
        self.solar_cell.reflection.load(self.prev_fullpath_refl)

        # EQE
        self.btn_EQE = QPushButton("Load EQE")
        self.btn_EQE.clicked.connect(self.get_EQE)
        grid.addWidget(self.btn_EQE, 2, 0)
        self.label_EQE = QLabel('example_EQE.txt', self)
        self.prev_fullpath_EQE = os.path.join(example_dir, 'example_EQE.txt')
        grid.addWidget(self.label_EQE, 2, 1)
        self.solar_cell.qe.load(self.prev_fullpath_EQE)

        # lightIV
        self.btn_lightIV = QPushButton("Load light IV")
        self.btn_lightIV.clicked.connect(self.get_lightIV)
        grid.addWidget(self.btn_lightIV, 3, 0)
        self.label_lightIV = QLabel('example_lightIV.lgt', self)
        self.prev_fullpath_lightIV = os.path.join(example_dir, 'example_lightIV.lgt')
        grid.addWidget(self.label_lightIV, 3, 1)
        self.solar_cell.liv.load(self.prev_fullpath_lightIV)
        # self.menu_lightIV = QComboBox()
        # self.menu_lightIV.addItems(['.lgt','.txt'])
        # grid.addWidget(self.menu_lightIV, 3, 2)

        # suns Voc
        self.btn_sunsVoc = QPushButton("Load suns Voc")
        self.btn_sunsVoc.clicked.connect(self.get_sunsVoc)
        grid.addWidget(self.btn_sunsVoc, 4, 0)
        self.label_sunsVoc = QLabel('example_sunsVoc.xlsm', self)
        self.prev_fullpath_sunsVoc = os.path.join(example_dir, 'example_sunsVoc.xlsm')
        grid.addWidget(self.label_sunsVoc, 4, 1)
        self.solar_cell.sunvoc.load(self.prev_fullpath_sunsVoc)

        # darkIV
        self.btn_darkIV = QPushButton("Load dark IV")
        self.btn_darkIV.clicked.connect(self.get_darkIV)
        grid.addWidget(self.btn_darkIV, 5, 0)
        self.label_darkIV = QLabel('example_darkIV.drk', self)
        self.prev_fullpath_darkIV = os.path.join(example_dir, 'example_darkIV.drk')
        grid.addWidget(self.label_darkIV, 5, 1)
        self.solar_cell.div.load(self.prev_fullpath_darkIV)

        # process all data
        self.btn_process = QPushButton("Process data")
        self.btn_process.clicked.connect(self.process_data)
        grid.addWidget(self.btn_process, 6, 0)

        self.setLayout(grid)
        self.setGeometry(100, 100, 300, 300)
        self.setWindowTitle('Loss analysis')
        self.show()

    def get_lightIV(self):
        default_dir = os.path.dirname(self.prev_fullpath_lightIV)
        full_path = QFileDialog.getOpenFileName(self, 'Choose light IV file',
                                                default_dir)[0]
        self.prev_fullpath_lightIV = full_path
        filename = os.path.basename(full_path)
        self.label_lightIV.setText(filename)
        self.solar_cell.liv.load(full_path)

    def get_darkIV(self):
        default_dir = os.path.dirname(self.prev_fullpath_darkIV)
        full_path = QFileDialog.getOpenFileName(self, 'Choose dark IV file',
                                                default_dir)[0]
        self.prev_fullpath_darkIV = full_path
        filename = os.path.basename(full_path)
        self.label_darkIV.setText(filename)
        self.solar_cell.div.load(full_path)

    def get_sunsVoc(self):
        default_dir = os.path.dirname(self.prev_fullpath_sunsVoc)
        full_path = QFileDialog.getOpenFileName(self, 'Choose dark IV file',
                                                default_dir)[0]
        self.prev_fullpath_sunsVoc = full_path
        filename = os.path.basename(full_path)
        self.label_sunsVoc.setText(filename)
        self.solar_cell.sunvoc.load(full_path)

    def get_refl(self):
        default_dir = os.path.dirname(self.prev_fullpath_refl)
        full_path = QFileDialog.getOpenFileName(self, 'Choose reflectance file',
                                                default_dir)[0]
        self.prev_fullpath_refl = full_path
        filename = os.path.basename(full_path)
        self.label_refl.setText(filename)
        self.solar_cell.reflection.load(full_path)

    def get_EQE(self):
        default_dir = os.path.dirname(self.prev_fullpath_EQE)
        full_path = QFileDialog.getOpenFileName(self, 'Choose EQE file',
                                                default_dir)[0]
        self.prev_fullpath_EQE = full_path
        filename = os.path.basename(full_path)
        self.label_EQE.setText(filename)
        self.solar_cell.qe.load(full_path)

    def process_data(self):
        self.solar_cell.process_all()

if __name__ == '__main__':

    logfile = open('traceback_log.txt','w')
    app = QApplication(sys.argv)
    # try:
    ex = Loss_analysis_gui()
    # except:
        # traceback.print_exc(file=logfile)

    ex.show()
    logfile.close()
    sys.exit(app.exec_())
