import os
import sys
from pathlib import Path
GUI_path = Path(os.path.abspath(os.path.dirname(__file__)))
ASG_path = GUI_path.parent.absolute()
sys.path.append(ASG_path)

import utils.PySimpleGUIQt as sg
from train_GUI import getTrainGUI, TrainhandleInput
from pred_GUI import getPredGUI, PredhandleInput
from eval_GUI import getEvalGUI, EvalhandleInput

is_supported = sys.version_info[:2] == (3, 10)
if sys.version_info[:2] != (3, 10):
    sg.popup(
        "⚠️ Python Version Warning",
        f"You are running Python {sys.version_info.major}.{sys.version_info.minor}\n\n"
        "This software was built and tested for Python 3.10.\n"
        "Other versions may not work correctly.\n\n"
        "Please install Python 3.10 if you encounter issues.",
        title="Version Warning",
        keep_on_top=True
    )
#from retrain_GUI import getRetrainGUI, RetrainhandleInput
working_directory=os.getcwd()

sg.set_options(font=("Arial Bold",14))

train_column = getTrainGUI()
pred_column = getPredGUI()
eval_column = getEvalGUI()
#retrain_column = getRetrainGUI()
layout = [[sg.TabGroup([
    [
        sg.Tab('Train', train_column),
        #sg.Tab('Retrain', retrain_column),
        sg.Tab('Prediction', pred_column),
        sg.Tab('Translation', eval_column)
    ]])],
]
window = sg.Window('BatSpot: The ANIMAL-SPOT based GUI', layout, finalize=True)
#sg.Print('All Re-routing complete, you can see what is happening in this Debug window. \n'
#         'Please do not close this Debug window until you are finished with this program', do_not_reroute_stdout=False)
while True:
   event, values = window.read()
   TrainhandleInput(event=event, values=values, window=window)
   #RetrainhandleInput(event=event, values=values, window=window)
   PredhandleInput(event=event, values=values, window=window)
   EvalhandleInput(event=event, values=values, window=window)
   print (event, values)
   if event in (sg.WIN_CLOSED, 'Exit'):
      break

window.close()