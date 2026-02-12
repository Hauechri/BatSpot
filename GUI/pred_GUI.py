import utils.PySimpleGUIQt as sg
import os
import logging
import threading
from pathlib import Path
import sys
from animal_spot.predict import start_predict, build_args
working_directory=os.getcwd()

sg.set_options(font=("Arial Bold",14))

#p_src_dir_label=sg.Text("ANIMAL-SPOT source directory:")
#p_src_dir_input=sg.InputText(key="-p_src_dir-")
#p_src_dir_filebrowser=sg.FolderBrowse(initial_folder=working_directory)

p_model_dir_label=sg.Text("Path model file:")
p_model_dir_input=sg.InputText(key="-p_model_dir-")
p_model_dir_filebrowser=sg.FileBrowse(initial_folder=working_directory)

p_log_dir_label=sg.Text("Path folder to store log:")
p_log_dir_input=sg.InputText(key="-p_log_dir-")
p_log_dir_filebrowser=sg.FolderBrowse(initial_folder=working_directory)

p_output_dir_label=sg.Text("Path folder to store output:")
p_output_dir_input=sg.InputText(key="-p_output_dir-")
p_output_dir_filebrowser=sg.FolderBrowse(initial_folder=working_directory)

p_input_file_label=sg.Text("Path folder with input files:")
p_input_file_input=sg.InputText(key="-p_input_file-")
p_input_file_filebrowser=sg.FileBrowse(initial_folder=working_directory)

p_debug_label=sg.Text("Enable debug:")
p_debug_checkbox=sg.Checkbox(text="", default=False, key="-p_debug-")

p_sequence_len_label=sg.Text("Prediction window size in ms:")
p_sequence_len_input=sg.InputText(key="-p_sequence_len-", default_text="300")
p_sequence_len_reset=sg.Button(button_text="default", key="p_default_sequence_len")

p_hop_label=sg.Text("Prediction hop size in ms:")
p_hop_input=sg.InputText(key="-p_hop-", default_text="100")
p_hop_reset=sg.Button(button_text="default", key="p_default_hop")

# p_threshold_label=sg.Text("Prediction threshold:")
# p_threshold_input=sg.InputText(key="-p_threshold-", default_text="0.75")
# p_threshold_reset=sg.Button(button_text="default", key="p_default_threshold")

p_no_cuda_label=sg.Text("Use cuda:")
p_no_cuda_checkbox=sg.Checkbox(text="", default=True, key="-p_no_cuda-")

p_visualize_label=sg.Text("Visualize output:")
p_visualize_checkbox=sg.Checkbox(text="", default=True, key="-p_visualize-")

p_jit_load_label=sg.Text("Use jit load:")
p_jit_load_checkbox=sg.Checkbox(text="", default=False, key="-p_jit_load-")

p_min_max_norm_label=sg.Text("Use min max normalization:")
p_min_max_norm_checkbox=sg.Checkbox(text="", default=True, key="-p_min_max_norm-")

p_latent_extract_label=sg.Text("Use latent extraction:")
p_latent_extract_checkbox=sg.Checkbox(text="", default=True, key="-p_latent_extract-")

p_batch_size_label=sg.Text("Batch size:")
p_batch_size_input=sg.InputText(key="-p_batch_size-", default_text="1")
p_batch_size_reset=sg.Button(button_text="default", key="p_default_batch_size")

p_num_workers_label=sg.Text("Number of worker:")
p_num_workers_input=sg.InputText(key="-p_num_workers-", default_text="0")
p_num_workers_reset=sg.Button(button_text="default", key="p_default_num_workers")

p_save_config_button=sg.FileSaveAs(button_text="Save settings")
p_save_config_Input=sg.Input(key="p_save_config", enable_events=True, visible=False)
p_load_config_button=sg.FileBrowse(button_text="Load settings")
p_load_config_Input=sg.Input(key="p_load_config", enable_events=True, visible=False)

p_start_prediction_button=sg.Button(button_text="Start prediction", key="p_start")
#p_output = sg.Output(size=(67, 10))

#sg.Print('Re-routing pred_GUI to Debug stdout', do_not_reroute_stdout=False)

pred_layout=[
    #[p_src_dir_label, p_src_dir_input, p_src_dir_filebrowser],
    [p_model_dir_label, p_model_dir_input, p_model_dir_filebrowser],
    [p_log_dir_label, p_log_dir_input, p_log_dir_filebrowser],
    [p_output_dir_label, p_output_dir_input, p_output_dir_filebrowser],
    [p_input_file_label, p_input_file_input, p_input_file_filebrowser],
    [p_debug_label, p_debug_checkbox],
    [p_sequence_len_label, p_sequence_len_input, p_sequence_len_reset],
    [p_hop_label, p_hop_input, p_hop_reset],
    # [p_threshold_label, p_threshold_input, p_threshold_reset],
    [p_no_cuda_label, p_no_cuda_checkbox],
    [p_visualize_label, p_visualize_checkbox],
    [p_jit_load_label, p_jit_load_checkbox],
    [p_min_max_norm_label, p_min_max_norm_checkbox],
    [p_latent_extract_label, p_latent_extract_checkbox],
    [p_batch_size_label, p_batch_size_input, p_batch_size_reset],
    [p_num_workers_label, p_num_workers_input, p_num_workers_reset],
    [p_save_config_button, p_save_config_Input],
    [p_load_config_button, p_load_config_Input],
    [p_start_prediction_button],
    #[p_output]

]

pred_column = [[sg.Column(pred_layout, scrollable=True, size=(1000,700))]]
def getPredGUI():
    return pred_column


def PredhandleInput(event, values, window):
    if event == "p_default_sequence_len":
        window['-p_sequence_len-'].update("300")
        values['-p_sequence_len-'] = "300"
    if event == "p_default_hop":
        window['-p_hop-'].update("100")
        values['-p_hop-'] = "100"
    if event == "p_default_threshold":
        window['-p_threshold-'].update("0.75")
        values['-p_threshold-'] = "0.75"
    if event == "p_default_batch_size":
        window['-p_batch_size-'].update("1")
        values['-p_batch_size-'] = "1"
    if event == "p_default_num_workers":
        window['-p_num_workers-'].update("0")
        values['-p_num_workers-'] = "0"
    if event == "p_save_config":
        generatePredConfig(values=values)
    if event == "p_load_config":
        loadPredConfig(values=values, window=window)
    if event == "p_start":
        startPrediction(values=values)


def generatePredConfig(values):
    file = open(p_save_config_Input.get(), "w")
    #Directorys
    #if values["-p_src_dir-"] == "":
    #    sg.popup_error("ANIMAL-SPOT source File not set")
    #    file.close()
    #    return
    #file.write("src_dir=" + str(values["-p_src_dir-"]) + "\n")

    if values["-p_model_dir-"] == "":
        sg.popup_error("Model directory not specified")
        file.close()
        return
    file.write("model_path=" + str(values["-p_model_dir-"]) + "\n")

    if values["-p_log_dir-"] == "":
        sg.popup_error("Checkpoint directory not specified")
        file.close()
        return
    file.write("log_dir=" + str(values["-p_log_dir-"]) + "\n")

    if values["-p_output_dir-"] == "":
        sg.popup_error("Output directory not specified")
        file.close()
        return
    file.write("output_dir=" + str(values["-p_output_dir-"]) + "\n")

    if values["-p_input_file-"] == "":
        sg.popup_error("Data directory not specified")
        file.close()
        return
    file.write("input_file=" + str(values["-p_input_file-"]) + "\n")

    #Boolean Parameter
    if values["-p_debug-"] is True:  # optional
        file.write("debug=" + str(values["-p_debug-"]) + "\n")

    if values["-p_no_cuda-"] == False:  # optional
        file.write("no_cuda=True" + "\n")
    else:
        file.write("no_cuda=False" + "\n")

    if values["-p_visualize-"] is True:  # optional
        file.write("visualize=" + str(values["-p_visualize-"]) + "\n")
    else:
        file.write("visualize=False" + "\n")

    if values["-p_jit_load-"] is True:  # optional
        file.write("jit_load=" + str(values["-p_jit_load-"]) + "\n")
    else:
        file.write("jit_load=False" + "\n")

    if values["-p_min_max_norm-"] is True:  # optional
        file.write("min_max_norm=" + str(values["-p_min_max_norm-"]) + "\n")
    else:
        file.write("min_max_norm=False" + "\n")

    if values["-p_latent_extract-"] is True:  # optional
        file.write("latent_extract=" + str(values["-p_latent_extract-"]) + "\n")
    else:
        file.write("latent_extract=False" + "\n")

    # Number Parameter
    if values["-p_sequence_len-"] != "":
        msvalue = float(values["-p_sequence_len-"])
        value = msvalue/1000.0
        file.write("sequence_len=" + str(value) + "\n")

    if values["-p_hop-"] != "":
        msvalue = float(values["-p_hop-"])
        value = msvalue/1000.0
        file.write("hop=" + str(values["-p_hop-"]) + "\n")

    if values["-p_threshold-"] != "":
        file.write("threshold=" + str(values["-p_threshold-"]) + "\n")

    if values["-p_batch_size-"] != "":
        file.write("batch_size=" + str(values["-p_batch_size-"]) + "\n")

    if values["-p_num_workers-"] != "":
        file.write("num_workers=" + str(values["-p_num_workers-"]) + "\n")
    file.close()

def loadPredConfig(values, window):
    file = open(p_load_config_Input.get())
    lines = file.readlines()
    for line in lines:
        if line.__contains__("#") or line.__contains__("*"):
            continue
        #if line.__contains__("src_dir="):
        #    val = line.split("=")[1]
        #    val = val.split("\n")[0]
        #    window['-p_src_dir-'].update(val)
        #    values['-p_src_dir-'] = val
        if line.__contains__("model_path="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-p_model_dir-'].update(val)
            values['-p_model_dir-'] = val
        if line.__contains__("log_dir="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-p_log_dir-'].update(val)
            values['-p_log_dir-'] = val
        if line.__contains__("output_dir="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-p_output_dir-'].update(val)
            values['-p_output_dir-'] = val
        if line.__contains__("input_file="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-p_input_file-'].update(val)
            values['-p_input_file-'] = val

        if line.__contains__("debug="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-p_debug-'].update(True)
                values['-p_debug-'] = True
            else:
                window['-p_debug-'].update(False)
                values['-p_debug-'] = False
        if line.__contains__("no_cuda="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-p_no_cuda-'].update(False)
                values['-p_no_cuda-'] = False
            else:
                window['-p_no_cuda-'].update(True)
                values['-p_no_cuda-'] = True
        if line.__contains__("visualize="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-p_visualize-'].update(True)
                values['-p_visualize-'] = True
            else:
                window['-p_visualize-'].update(False)
                values['-p_visualize-'] = False
        if line.__contains__("jit_load="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-p_jit_load-'].update(True)
                values['-p_jit_load-'] = True
            else:
                window['-p_jit_load-'].update(False)
                values['-p_jit_load-'] = False
        if line.__contains__("min_max_norm="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-p_min_max_norm-'].update(True)
                values['-p_min_max_norm-'] = True
            else:
                window['-p_min_max_norm-'].update(False)
                values['-p_min_max_norm-'] = False
        if line.__contains__("latent_extract="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-p_latent_extract-'].update(True)
                values['-p_latent_extract-'] = True
            else:
                window['-p_latent_extract-'].update(False)
                values['-p_latent_extract-'] = False

        if line.__contains__("sequence_len="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            value = float(val) * 1000.0
            value = str(int(value))
            window['-p_sequence_len-'].update(value)
            values['-p_sequence_len-'] = value
        if line.__contains__("hop="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            #transform s to ms
            val = float(val)*1000
            window['-p_hop-'].update(str(val))
            values['-p_hop-'] = val
        if line.__contains__("threshold="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-p_threshold-'].update(val)
            values['-p_threshold-'] = val
        if line.__contains__("batch_size="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-p_batch_size-'].update(val)
            values['-p_batch_size-'] = val
        if line.__contains__("num_workers="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-p_num_workers-'].update(val)
            values['-p_num_workers-'] = val
    file.close()

def startPrediction_old(values):
    GUI_path = Path(os.path.abspath(os.path.dirname(__file__)))
    ASG_path = GUI_path.parent.absolute()
    pythonexe = 'python'#os.path.join(ASG_path, 'venv/Scripts/python.exe')
    pred_cmd = pythonexe + " -W ignore::UserWarning"

    #Directorys
    #if values["-p_src_dir-"] == "":
    #    sg.popup_error("ANIMAL-SPOT source File not set")
    #    return
    #elif not os.path.isfile(values["-p_src_dir-"] + "/predict.py"):
    #    sg.popup_error("Source File error")
    #    return
    #pred_cmd = pred_cmd + " " + values["-p_src_dir-"] + "/predict.py"

    if values["-p_model_dir-"] == "":
        sg.popup_error("Model directory not specified")
        return
    pred_cmd = pred_cmd + " --model_path " + values["-p_model_dir-"]

    if values["-p_log_dir-"] == "":
        sg.popup_error("Checkpoint directory not specified")
        return
    pred_cmd = pred_cmd + " --log_dir " + values["-p_log_dir-"]

    if values["-p_output_dir-"] == "":
        sg.popup_error("Output directory not specified")
        return
    pred_cmd = pred_cmd + " --output_dir " + values["-p_output_dir-"]

    if values["-p_input_file-"] == "":
        sg.popup_error("Data directory not specified")
        return
    pred_cmd = pred_cmd + " --input_file " + values["-p_input_file-"]

    #Boolean Parameter
    if values["-p_debug-"] is True:  # optional
        pred_cmd = pred_cmd + " --debug"

    if values["-p_no_cuda-"] == False:  # optional
        pred_cmd = pred_cmd + " --no_cuda"

    if values["-p_visualize-"] is True:  # optional
        pred_cmd = pred_cmd + " --visualize"

    if values["-p_jit_load-"] is True:  # optional
        pred_cmd = pred_cmd + " --jit_load"

    if values["-p_min_max_norm-"] is True:  # optional
        pred_cmd = pred_cmd + " --min_max_norm"

    if values["-p_latent_extract-"] is True:  # optional
        pred_cmd = pred_cmd + " --latent_extract"

    # Number Parameter
    if values["-p_sequence_len-"] != "":
        msvalue = float(values["-p_sequence_len-"])
        value = msvalue/1000.0
        pred_cmd = pred_cmd + " --sequence_len " + str(value)

    if values["-p_hop-"] != "":
        #hop ms to s
        pred_cmd = pred_cmd + " --hop " + str(float(values["-p_hop-"])/1000.0)

    if values["-p_threshold-"] != "":
        pred_cmd = pred_cmd + " --threshold " + values["-p_threshold-"]

    if values["-p_batch_size-"] != "":
        pred_cmd = pred_cmd + " --batch_size " + values["-p_batch_size-"]

    if values["-p_num_workers-"] != "":
        pred_cmd = pred_cmd + " --num_workers " + values["-p_num_workers-"]

    t1 = threading.Thread(target=startPredCommand, args=(pred_cmd,))
    t1.start()
    sg.popup('The Prediction has started in Thread ' + str(t1.ident) +
             '\nPlease look at the console to follow the prediction progress or errors. '
             'Also please do not start another Thread unless you know what you are doing!')

def startPredCommand(pred_cmd):
    logger = logging.getLogger('training animal-spot')
    stream_handler = logging.StreamHandler()
    logger.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.info("Prediction Command: " + pred_cmd)
    logger.info("Start Prediction!!!")
    os.system(pred_cmd)

def gui_values_to_predict_arglist(values):
    args = []

    def add(flag, key):
        if values[key] != "":
            args.extend([flag, str(values[key])])

    def add_bool(flag, key):
        if values[key] is True:
            args.append(flag)

    # required
    add("--model_path", "-p_model_dir-")
    add("--log_dir", "-p_log_dir-")
    add("--output_dir", "-p_output_dir-")
    add("--input_file", "-p_input_file-")

    # booleans
    add_bool("--debug", "-p_debug-")
    add_bool("--visualize", "-p_visualize-")
    add_bool("--jit_load", "-p_jit_load-")
    add_bool("--min_max_norm", "-p_min_max_norm-")
    add_bool("--latent_extract", "-p_latent_extract-")

    # inverted CUDA flag
    if values["-p_no_cuda-"] is False:
        args.append("--no_cuda")

    # numeric (GUI gives ms → convert to seconds)
    if values["-p_sequence_len-"] != "":
        args.extend(["--sequence_len", str(float(values["-p_sequence_len-"]) / 1000.0)])

    if values["-p_hop-"] != "":
        args.extend(["--hop", str(float(values["-p_hop-"]) / 1000.0)])

    add("--threshold", "-p_threshold-")
    add("--batch_size", "-p_batch_size-")
    add("--num_workers", "-p_num_workers-")

    return args

def startPrediction(values):
    # basic validation (keep your popups)
    if values["-p_model_dir-"] == "":
        sg.popup_error("Model directory not specified")
        return
    if values["-p_input_file-"] == "":
        sg.popup_error("Input file not specified")
        return
    if values["-p_output_dir-"] == "":
        sg.popup_error("Output directory not specified")
        return

    arg_list = gui_values_to_predict_arglist(values)

    def run():
        ARGS = build_args(arg_list)
        start_predict(ARGS)

    t1 = threading.Thread(target=run, daemon=True)
    t1.start()

    sg.popup(
        'The Prediction has started in Thread ' + str(t1.ident) +
        '\nPlease look at the console to follow the prediction progress or errors.'
    )