import sys
import traceback
import os
from PyQt5.QtWidgets import (QWidget, QFileDialog, QPushButton, QTextEdit,
                             QGridLayout, QApplication, QLabel, QComboBox,
                             QCheckBox, QLineEdit, QStatusBar, QMainWindow)
from PyQt5.QtCore import pyqtSignal
# files for this package
import loss_analysis


class Measurement(QWidget):

    def __init__(self, grid, meas_name, default_file, row, show_term):
        super().__init__()

        self.meas_name = meas_name
        # this is used to determine if a item should be shown
        self.show_term = show_term

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

    def visability(self, term_dic):

        self.btn.hide()
        self.label.hide()
        for term in self.show_term:
            if term_dic[term]:
                self.btn.show()
                self.label.show()


class LoadFiles(QWidget):

    hide_panel = pyqtSignal()

    selection = {'Jsc': False,
                 'Voc': False,
                 'FF': False
                 }

    boxes = [
        ['reflectance', 'example_reflectance.csv', ['Jsc']],
        ['EQE', 'example_EQE.txt', ['Jsc']],
        ['light IV', 'example_lightIV.lgt', ['Voc', 'FF']],
        ['suns Voc', 'example_sunsVoc.xlsm', ['FF']],
        ['dark IV', 'example_darkIV.drk', ['FF']]
    ]

    def __init__(self, parent):

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
        # grid.addWidget(self.btn_start_dir, 0, 0)
        # self.label_start_dir = QLabel(os.path.basename(self.start_dir), self)
        # grid.addWidget(self.label_start_dir, 0, 1)

        # select output directory
        self.btn_output_dir = QPushButton("Select output directory")
        self.btn_output_dir.clicked.connect(self.select_output_dir)
        # grid.addWidget(self.btn_output_dir, 1, 0)
        # self.label_output_dir = QLabel(os.path.basename(self.output_dir), self)
        # grid.addWidget(self.label_output_dir, 1, 1)

        self.measurement = []

        for row_num, box in enumerate(self.boxes):
            self.measurement.append(Measurement(grid, box[0], box[1],
                                                row_num, box[2]))

        # save figures checkbox
        # self.cb_save_fig = QCheckBox('Save figures', self)
        # self.cb_save_fig.stateChanged.connect(self.save_fig_toggle)
        # grid.addWidget(self.cb_save_fig, 9, 0)

        # cell name input
        # self.cell_name_input = QLineEdit(self)
        # self.cell_name_input.setPlaceholderText('cell name')
        # grid.addWidget(self.cell_name_input, 9, 1)

        # process all data
        self.btn_process = QPushButton("Process data")
        self.btn_process.clicked.connect(self.process_data)
        grid.addWidget(self.btn_process, row_num + 1, 1)

        self.back = QPushButton("Back")
        self.back.clicked.connect(self.hide_widget)
        grid.addWidget(self.back, row_num + 1, 0)

        self.check_visability()

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

    def check_visability(self):
        self.parent.statusBar().showMessage('Please select the files')
        for box in self.measurement:
            box.visability(self.selection)

        self.show()

    def hide_widget(self):
        self.hide()
        self.hide_panel.emit()

    def process_data(self):
        example_dir = os.path.join(os.path.dirname(
            __file__), os.pardir, 'example_cell')
        print(example_dir, '\n\n\n\n\n', os.path.dirname(
            __file__))
        files = {
            'reflectance_fname': os.path.join(example_dir, 'example_reflectance.csv'),
            'EQE_fname': os.path.join(example_dir, 'example_EQE.txt'),
            'light IV_fname': os.path.join(example_dir, 'example_lightIV.lgt'),
            'suns Voc_fname': os.path.join(example_dir, 'example_sunsVoc.xlsm'),
            'dark IV_fname': os.path.join(example_dir, 'example_darkIV.drk')}

        cell1 = loss_analysis.Cell(**files)
        cell1.process_all(False, 'test', 'test')
        pass


class LossOptions(QWidget):

    hide_panel = pyqtSignal()

    def __init__(self, parent):
        # super(LossAnalysisGui, self).__init__(parent)
        super().__init__()
        self.parent = parent

        self.initUI()

    def initUI(self):

        btns = []
        labs = ['Jsc loss', 'Voc loss', 'FF loss', 'All losses']
        for lab in labs:
            btns.append(QPushButton(lab))
            btns[-1].clicked.connect(self.loss)
        grid = QGridLayout()
        grid.setSpacing(1)

        for i, btn in enumerate(btns):
            grid.addWidget(btn, i, 0)

        self.setLayout(grid)
        self.show()

    def _clear(self):
        for key in LoadFiles.selection.keys():
            LoadFiles.selection[key] = False

    def loss(self):

        # get the text from the sender
        selection = self.sender().text().split(' ')[0].lower()
        # clears the diction that controls what is hidden
        self._clear()
        # applies the required values to the dictionary
        if selection == 'all':
            self._all()
        else:
            selection = self.sender().text().split(' ')[0]
            LoadFiles.selection[selection] = True

        self.hide_panel.emit()
        self.hide()

    def _all(self):
        for key in LoadFiles.selection.keys():
            LoadFiles.selection[key] = True

    def _show(self):
        self.parent.statusBar().showMessage('Please select the Loss analysis')
        self.show()


class LossAnalysisGui(QWidget):

    def __init__(self, parent):
        # super(LossAnalysisGui, self).__init__(parent)
        super().__init__()
        self.parent = parent

        self.initUI()

    def initUI(self):

        grid = QGridLayout()
        lo = LossOptions(self.parent)
        la = LoadFiles(self.parent)

        grid.addWidget(lo, 0, 0)
        grid.addWidget(la, 0, 1)

        lo.hide_panel.connect(la.check_visability)
        la.hide_panel.connect(lo._show)

        self.setLayout(grid)

        # q hack so the window size doesn't change
        la.hide()
        lo._show()
        self.show()


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Loss analysis'
        self.left = 100
        self.top = 100
        self.width = 300
        self.height = 200
        self.initUI()

    def initUI(self):

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.statusBar().showMessage('Please select the Loss analysis')

        grid = QGridLayout()

        self.form_widget = LossAnalysisGui(self)
        self.setCentralWidget(self.form_widget)

        self.show()


if __name__ == '__main__':

    logfile = open('traceback_log.txt', 'w')
    app = QApplication(sys.argv)
    # try:
    lag = App()
    # except:
    # traceback.print_exc(file=logfile)

    lag.show()
    logfile.close()
    sys.exit(app.exec_())
