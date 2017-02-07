import sys
import traceback
import os
from PyQt5.QtWidgets import (QWidget, QFileDialog, QPushButton, QTextEdit,
                             QGridLayout, QApplication, QLabel, QComboBox)
# files for this package
import loss_analysis


class LoadButtonCombo(QWidget):
#TODO: couldn't this whole class just be a single function?

    def __init__(self, grid, info, default_file, row, column):
        super().__init__()

        self.info = info

        path = os.sep.join(os.path.dirname(
            os.path.realpath(__file__)).split(os.sep)[:-1])

        self.example_dir = os.path.join(path, 'example_cell')
        self.filepath = os.path.join(self.example_dir, default_file)

        self._add_objects(grid, row, column)

    def _add_objects(self, grid,  row, column):
        '''
        Builds and binds the boxes.
        '''
        self.btn = QPushButton('Load {0}'.format(self.info))
        self.btn.clicked.connect(self._get)

        filename = os.path.basename(self.filepath)
        self.label = QLabel(filename, self)

        grid.addWidget(self.btn, row, column)
        grid.addWidget(self.label, row, column + 1)

    def _get(self):
        '''
        Gets and sets the label with the new file name
        '''
        default_dir = os.path.dirname(self.filepath)
        self.filepath = QFileDialog.getOpenFileName(self, 'Choose {0} file'.format(self.info),
                                                    default_dir)[0]
        filename = os.path.basename(self.filepath)
        self.label.setText(filename)

    def file(self):
        return {self.info + '_fname': self.filepath}


class LossAnalysisGui(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        # print(path, example_dir)
        grid = QGridLayout()
        # grid.setSpacing(10)

        boxes = [['reflectance', 'example_reflectance.csv'],
                 ['EQE', 'example_EQE.txt'],
                 ['light IV', 'example_lightIV.lgt'],
                 ['suns Voc', 'example_sunsVoc.xlsm'],
                 ['dark IV', 'example_darkIV.drk']
                 ]

        self.items = []

        # TODO Ned: I'm not convinced this is the best method
        for box, row_num in zip(boxes, range(len(boxes))):
            self.items.append(LoadButtonCombo(grid, box[0], box[1],
                                              row_num + 1, 0))

        # process all data
        self.btn_process = QPushButton("Process data")
        self.btn_process.clicked.connect(self.process_data)
        grid.addWidget(self.btn_process, 6, 0)

        self.setLayout(grid)
        self.setGeometry(100, 100, 300, 300)
        self.setWindowTitle('Loss analysis')
        self.show()

    def process_data(self):

        files = {}
        for i in self.items:
            files.update(i.file())

        # pass the file names, and let the next thing handle them.
        la = loss_analysis.LossAnalysisHandler(**files)
        la.process_all()


if __name__ == '__main__':

    logfile = open('traceback_log.txt', 'w')
    app = QApplication(sys.argv)
    # try:
    ex = LossAnalysisGui()
    # except:
    # traceback.print_exc(file=logfile)

    ex.show()
    logfile.close()
    sys.exit(app.exec_())
