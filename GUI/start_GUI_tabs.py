import os
import sys
import multiprocessing
from pathlib import Path

GUI_path = Path(os.path.abspath(os.path.dirname(__file__)))
ASG_path = GUI_path.parent.absolute()
sys.path.append(ASG_path)

import utils.PySimpleGUIQt as sg
from train_GUI import getTrainGUI, TrainhandleInput
from pred_GUI import getPredGUI, PredhandleInput
from eval_GUI import getEvalGUI, EvalhandleInput


def main():

    if sys.version_info[:2] != (3, 10):
        sg.popup(
            "Python Version Warning",
            f"You are running Python {sys.version_info.major}.{sys.version_info.minor}\n\n"
            "This software was built and tested for Python 3.10.\n"
            "Other versions may not work correctly.\n\n"
            "Please install Python 3.10 if you encounter issues.",
            title="Version Warning",
            keep_on_top=True
        )

    sg.set_options(font=("Arial Bold", 14))

    train_column = getTrainGUI()
    pred_column = getPredGUI()
    eval_column = getEvalGUI()

    layout = [[
        sg.TabGroup([[
            sg.Tab('Train', train_column),
            sg.Tab('Prediction', pred_column),
            sg.Tab('Translation', eval_column)
        ]])
    ]]

    window = sg.Window(
        'BatSpot: The ANIMAL-SPOT based GUI',
        layout,
        finalize=True
    )

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, 'Exit'):
            break

        TrainhandleInput(event=event, values=values, window=window)
        PredhandleInput(event=event, values=values, window=window)
        EvalhandleInput(event=event, values=values, window=window)

        print(event, values)

    window.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()