import sys
import traceback
import os
from PyQt5.QtWidgets import (QWidget, QFileDialog, QPushButton, QTextEdit,
                             QGridLayout, QApplication, QLabel, QComboBox,
                             QCheckBox, QLineEdit, QStatusBar, QMainWindow)
# files for this package
import loss_analysis


class Measurement(QWidget):

    def __init__(self, grid, meas_name, default_file, row):
        super().__init__()

        self.meas_name = meas_name

        self.start_dir = os.path.join(os.pardir, 'example_cell')
        self.filepath = os.path.join(self.start_dir, default_file)

        _, self.file_ext = os.path.splitext(default_file)

        self._add_objects(grid, row)

    def _add_objects(self, grid,  row):
        '''
        Builds and binds the boxes.
        '''
        self.btn = QPushButton('Load {0}'.format(self.meas_name))
        self.btn.clicked.connect(self._get)

        filename = os.path.basename(self.filepath)

        self.label = QLabel(filename, self)
        grid.addWidget(self.btn, row, 0)
        grid.addWidget(self.label, row, 1)

    def _get(self):
        '''
        Gets and sets the label with the new file name
        '''
        filter_str = '{0} file (*{1})'.format(self.meas_name, self.file_ext)
        self.filepath = QFileDialog.getOpenFileName(self,
                                                    'Choose {0} file'.format(
                                                        self.meas_name),
                                                    self.start_dir, filter_str)[0]
        filename = os.path.basename(self.filepath)
        self.label.setText(filename)

    def file(self):
        return {self.meas_name + '_fname': self.filepath}


class LossAnalysisGui(QWidget):

    def __init__(self, parent):
        # super(LossAnalysisGui, self).__init__(parent)
        super().__init__()
        self.parent = parent

        self.initUI()

    def initUI(self):

        grid = QGridLayout()
        # grid.setSpacing(10)
        self.output_dir = os.path.join(os.pardir, 'example_cell')
        self.start_dir = os.path.join(os.pardir, 'example_cell')
        self.save_fig_bool = False

        # select starting directory
        self.btn_start_dir = QPushButton("Select start directory")
        self.btn_start_dir.clicked.connect(self.select_start_dir)
        grid.addWidget(self.btn_start_dir, 0, 0)
        self.label_start_dir = QLabel(os.path.basename(self.start_dir), self)
        grid.addWidget(self.label_start_dir, 0, 1)

        # select output directory
        self.btn_output_dir = QPushButton("Select output directory")
        self.btn_output_dir.clicked.connect(self.select_output_dir)
        grid.addWidget(self.btn_output_dir, 1, 0)
        self.label_output_dir = QLabel(os.path.basename(self.output_dir), self)
        grid.addWidget(self.label_output_dir, 1, 1)

        boxes = [['reflectance', 'example_reflectance.csv'],
                 ['EQE', 'example_EQE.txt'],
                 ['light IV', 'example_lightIV.lgt'],
                 ['suns Voc', 'example_sunsVoc.xlsm'],
                 ['dark IV', 'example_darkIV.drk']
                 ]

        self.measurement = []

        for box, row_num in zip(boxes, range(len(boxes))):
            self.measurement.append(Measurement(grid, box[0], box[1],
                                                row_num + 3))

        # save figures checkbox
        self.cb_save_fig = QCheckBox('Save figures', self)
        self.cb_save_fig.stateChanged.connect(self.save_fig_toggle)
        grid.addWidget(self.cb_save_fig, 9, 0)

        # cell name input
        self.cell_name_input = QLineEdit(self)
        self.cell_name_input.setPlaceholderText('cell name')
        grid.addWidget(self.cell_name_input, 9, 1)

        # process all data
        self.btn_process = QPushButton("Process data")
        self.btn_process.clicked.connect(self.process_data)
        grid.addWidget(self.btn_process, 10, 0)

        # self.statusBar = QStatusBar()
        # self.setStatusBar(self.statusBar)

        self.setLayout(grid)

        self.show()

    def select_start_dir(self):
        self.start_dir = QFileDialog.getExistingDirectory(self,
                                                          'Choose start directory', self.start_dir)
        self.label_start_dir.setText(os.path.basename(self.start_dir))
        for m in self.measurement:
            m.start_dir = self.start_dir

        self.output_dir = self.start_dir
        self.label_output_dir.setText(os.path.basename(self.start_dir))

    def select_output_dir(self):
        self.output_dir = QFileDialog.getExistingDirectory(self,
                                                           'Choose output directory', self.output_dir)
        self.label_output_dir.setText(os.path.basename(self.output_dir))

    def save_fig_toggle(self, state):
        if self.cb_save_fig.isChecked():
            self.save_fig_bool = True
        else:
            self.save_fig_bool = False

    def process_data(self):

        files = {}
        for i in self.measurement:
            files.update(i.file())

        # pass the file names, and let the next thing handle them.
        self.parent.statusBar().showMessage('loading files')

        # a check on the data
        # if the data is bad, a message is returned in the gui
        try:
            la = loss_analysis.Cell(**files)
        except Exception as e:
            self.parent.statusBar().showMessage('Error:' + str(e))
            # print(str(e))
        else:
            self.parent.statusBar().showMessage('Calculating losses')

        la.process_all(self.save_fig_bool, self.output_dir,
                       self.cell_name_input.text())
        self.parent.statusBar().showMessage('Done!')


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Loss analysis'
        self.left = 100
        self.top = 100
        self.width = 400
        self.height = 500
        self.initUI()

    def initUI(self):

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.statusBar().showMessage('All clear, ready to roll')

        self.form_widget = LossAnalysisGui(self)
        self.setCentralWidget(self.form_widget)

        self.show()


if __name__ == '__main__':

    # logfile = open('traceback.log', 'w')
    app = QApplication(sys.argv)
    # try:
    lag = App()
    # except:
    # traceback.print_exc(file=logfile)

    lag.show()
    # logfile.close()
    sys.exit(app.exec_())
