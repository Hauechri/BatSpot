import utils.PySimpleGUIQt as sg
import os
from pathlib import Path
import sys
working_directory=os.getcwd()

sg.set_options(font=("Arial Bold",14))

e_src_dir_label=sg.Text("Path folder prediction output:")
e_src_dir_input=sg.InputText(key="-e_prediction_dir-")
e_src_dir_filebrowser=sg.FolderBrowse(initial_folder=working_directory)

e_output_dir_label=sg.Text("Path folder to store output:")
e_output_dir_input=sg.InputText(key="-e_output_dir-")
e_output_dir_filebrowser=sg.FolderBrowse(initial_folder=working_directory)

e_noise_in_anno_label=sg.Text("Noise in annotation:")
e_noise_in_anno_checkbox=sg.Checkbox(text="", default=False, key="-e_noise_in_anno-")

e_threshold_label=sg.Text("Prediction threshold:")
e_threshold_input=sg.InputText(key="-e_threshold-", default_text="0.85")
e_threshold_reset=sg.Button(button_text="default", key="e_default_threshold")

e_save_config_button=sg.FileSaveAs(button_text="Save settings")
e_save_config_Input=sg.Input(key="e_save_config", enable_events=True, visible=False)
e_load_config_button=sg.FileBrowse(button_text="Load settings")
e_load_config_Input=sg.Input(key="e_load_config", enable_events=True, visible=False)

e_start_prediction_button=sg.Button(button_text="Start evaluation", key="e_start")
#e_output = sg.Output(size=(67, 10))

#sg.Print('Re-routing eval_GUI to Debug stdout', do_not_reroute_stdout=False)


eval_layout=[
    [e_src_dir_label, e_src_dir_input, e_src_dir_filebrowser],
    [e_output_dir_label, e_output_dir_input, e_output_dir_filebrowser],
    [e_noise_in_anno_label, e_noise_in_anno_checkbox],
    [e_threshold_label, e_threshold_input, e_threshold_reset],
    [e_save_config_button, e_save_config_Input],
    [e_load_config_button, e_load_config_Input],
    [e_start_prediction_button],
    #[e_output]
]
eval_column = [[sg.Column(eval_layout, scrollable=True, size=(1000,700))]]
def getEvalGUI():
    return eval_column

def EvalhandleInput(event, values, window):
    if event == "e_default_threshold":
        window['-e_threshold-'].update("0.85")
        values['-e_threshold-'] = "0.85"
    if event == "e_save_config":
        generateEvalConfig(values=values)
    if event == "e_load_config":
        loadEvalConfig(values=values, window=window)
    if event == "e_start":
        startEvaluation(values=values)

def generateEvalConfig(values):
    file = open(e_save_config_Input.get(), "w")
    if values["-e_prediction_dir-"] == "":
        sg.popup_error("Prediction directory not set")
        file.close()
        return
    file.write("prediction_dir="+str(values["-e_prediction_dir-"])+"\n")

    if values["-e_output_dir-"] == "":
        sg.popup_error("Output directory not set")
        file.close()
        return
    file.write("output_dir="+str(values["-e_output_dir-"])+"\n")

    if values["-e_threshold-"] == "":
        sg.popup_error("Threshold not set")
        file.close()
        return
    file.write("threshold="+str(values["-e_threshold-"])+"\n")
    if values["-e_noise_in_anno-"] != "":
        file.write("noise_in_anno="+str(values["-e_noise_in_anno-"]))
    file.close()

def loadEvalConfig(values, window):
    file = open(e_load_config_Input.get())
    lines = file.readlines()
    for line in lines:
        if line.__contains__("#") or line.__contains__("*"):
            continue
        if line.__contains__("prediction_dir="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-e_prediction_dir-'].update(val)
            values['-e_prediction_dir-'] = val
        if line.__contains__("output_dir="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-e_output_dir-'].update(val)
            values['-e_output_dir-'] = val
        if line.__contains__("threshold="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-e_threshold-'].update(val)
            values['-e_threshold-'] = val
        if line.__contains__("noise_in_anno="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-e_noise_in_anno-'].update(True)
                values['-e_noise_in_anno-'] = True
            else:
                window['-e_noise_in_anno-'].update(False)
                values['-e_noise_in_anno-'] = False
    file.close()

def getEvalConfig(values):
    config_data = dict()
    if values["-e_prediction_dir-"] == "":
        sg.popup_error("Prediction directory not set")
        return
    config_data["prediction_dir"] = values["-e_prediction_dir-"]
    if values["-e_output_dir-"] == "":
        sg.popup_error("Output directory not set")
        return
    config_data["output_dir"] = values["-e_output_dir-"]
    if values["-e_threshold-"] == "":
        sg.popup_error("Threshold not set")
        return
    config_data["threshold"] = values["-e_threshold-"]
    if values["-e_noise_in_anno-"] == "true":
        config_data["noise_in_anno"] = ""
    return config_data


import ast

import logging
import platform

from collections import OrderedDict


def get_dict_key_ignorecase(dict_in, key_t):
    dict_lc = dict()
    for key in dict_in:
        dict_lc[key.lower()] = dict_in.get(key)
    key_t = key_t.lower()
    if dict_lc.get(key_t) is not None:
        return dict_lc.get(key_t)
    else:
        return None

class setup_evaluator(object):

    def __init__(self, config, log_level="error"):
        self.logger = None
        self.config_path = config
        self.config_data = dict()
        self.log_level = log_level
        self.loglevels = ["debug", "error", "warning", "info"]
        self.classes_label_idx = None
        self.classes_idx_label = None
        self.duration = None
        self.sequence_len = None
        self.hop = None
        self.data_opts = None
        self.needed_annotation_columns = ["Selection", "View", "Channel", "Begin time (s)", "End time (s)",
                                          "Low Freq (Hz)", "High Freq (Hz)", "Sound type", "Comments"]

    def init_logger(self):
        self.logger = logging.getLogger('training animal-spot')
        stream_handler = logging.StreamHandler()
        if self.log_level.lower() == self.loglevels[0]:
            self.logger.setLevel(logging.DEBUG)
            stream_handler.setLevel(logging.DEBUG)
        elif self.log_level.lower() == self.loglevels[1]:
            self.logger.setLevel(logging.ERROR)
            stream_handler.setLevel(logging.ERROR)
        elif self.log_level.lower() == self.loglevels[2]:
            self.logger.setLevel(logging.WARNING)
            stream_handler.setLevel(logging.WARNING)
        elif self.log_level.lower() == self.loglevels[3]:
            self.logger.setLevel(logging.INFO)
            stream_handler.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.DEBUG)
            stream_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def read_config(self):
        config_file = open(self.config_path, 'r')
        for element in config_file:
            if element[0] != "#" and element[0] != " " and element[0] != "" and element[0] != "\n":
                key = element.strip().split("=")[0]
                value = element.strip().split("=")[1]
                if value == "true":
                    self.config_data[key] = ""
                elif value == "false":
                    continue
                else:
                    self.config_data[key] = value
        self.logger.info("Config Data: " + str(self.config_data))

    def read_prediction_file(self, path_prediction_file):
        time_prob = []
        prediction_file = open(path_prediction_file, "r")
        for line in prediction_file:
            print(line)
            if line.find("pred=") != -1 and line.find("prob=") != -1:
                if len(self.classes_idx_label) <= 2:
                    if line.strip().find("|INFO|") != -1:
                        self.logger.debug(line.strip().split("|INFO|")[1])
                        time_prob.append((line.strip().split("|INFO|")[1].split(",")[0].strip().split("time=")[1],
                                          line.strip().split("|INFO|")[1].split("prob=")[1].strip(), None))
                        self.duration = \
                        line.strip().split("|INFO|")[1].split(",")[0].strip().split("time=")[1].split("-")[1]
                    else:
                        self.logger.debug(line.strip().split("|I|")[1])
                        time_prob.append((line.strip().split("|I|")[1].split(",")[0].strip().split("time=")[1],
                                          line.strip().split("|I|")[1].split("prob=")[1].strip(), None))
                        self.duration = line.strip().split("|I|")[1].split(",")[0].strip().split("time=")[1].split("-")[
                            1]
                else:
                    if line.strip().find("|INFO|") != -1:
                        self.logger.debug(line.strip().split("|INFO|")[1])
                        time_prob.append((line.strip().split("|INFO|")[1].split(",")[0].strip().split("time=")[1],
                                          line.strip().split("|INFO|")[1].split("prob=")[1].strip(),
                                          line.strip().split("|INFO|")[1].split("pred_class=")[1].strip().split(",")[
                                              0].strip().split("-")[1].strip()))
                        self.duration = \
                        line.strip().split("|INFO|")[1].split(",")[0].strip().split("time=")[1].split("-")[1]
                    else:
                        self.logger.debug(line.strip().split("|I|")[1])
                        time_prob.append((line.strip().split("|I|")[1].split(",")[0].strip().split("time=")[1],
                                          line.strip().split("|I|")[1].split("prob=")[1].strip(),
                                          line.strip().split("|I|")[1].split("pred_class=")[1].strip().split(",")[
                                              0].strip().split("-")[1].strip()))
                        self.duration = line.strip().split("|I|")[1].split(",")[0].strip().split("time=")[1].split("-")[
                            1]
            elif line.find("prediction length") != -1 and line.find("hop length") != -1:
                self.sequence_len = line.split(",")[0].split(" ")[-1].split("[")[0]
                self.hop = line.split(",")[1].split(" ")[-1].split("[")[0]
            elif line.find("class distribution") != -1:
                self.classes_label_idx = ast.literal_eval(line.split("class distribution:")[-1].strip())
                self.classes_idx_label = dict(zip(self.classes_label_idx.values(), self.classes_label_idx.keys()))
            elif line.find("dataOpts") != -1:
                self.data_opts = ast.literal_eval(line.split("dataOpts:")[-1].strip())
        self.logger.info("duration of tape=" + self.duration)
        self.logger.info("prediction file has been processed successfully")
        return time_prob

    def get_all_network_prediction_data(self, time_prob, threshold):
        threshold_depended_pred_data = OrderedDict()
        for time, prob, pred_class_idx in time_prob:
            if pred_class_idx is None:
                if float(prob) >= float(threshold):
                    threshold_depended_pred_data[time] = [1, str(prob)]
                    self.logger.debug("time=" + time + ", pred=1, prob=" + str(prob) + "\n")
                else:
                    threshold_depended_pred_data[time] = [0, str(prob)]
                    self.logger.debug("time=" + time + ", pred=0, prob=" + str(prob) + "\n")
            else:
                if float(prob) >= float(threshold):
                    threshold_depended_pred_data[time] = [pred_class_idx, str(prob)]
                    self.logger.debug("time=" + time + ", pred=1, prob=" + str(prob) + "\n")
                else:
                    threshold_depended_pred_data[time] = [str(get_dict_key_ignorecase(self.classes_label_idx, "noise")),
                                                          str(prob)]
                    self.logger.debug("time=" + time + ", pred=0, prob=" + str(prob) + "\n")
        labeled_data, duration = self.get_time_info(threshold_depended_pred_data)
        return labeled_data, duration

    def overlapped_times_to_pred(self, time_info, prediction_data):
        overlapped_pred_data = OrderedDict()
        for timestamp in prediction_data:
            overlapped_pred_data[timestamp] = [0, prediction_data.get(timestamp)[1]]
        for time, label in time_info:
            timestamps = self.generate_timestamps(time[0], time[1], round(float(self.sequence_len), 3),
                                                  round(float(self.hop), 3))
            for update_timestamp in timestamps:
                overlapped_pred_data[update_timestamp] = [1, overlapped_pred_data.get(update_timestamp)[1]]
        return overlapped_pred_data

    def generate_timestamps(self, start, end, sequence_len, hop_size, human=False):
        if round(start, 3) == round(end, 3):
            return []
        else:
            if human:
                start = start - sequence_len + hop_size
                end = end + sequence_len
                if start < 0:
                    start = 0
                if end > round(float(self.duration), 3):
                    end = round(float(self.duration), 3)
            tmp_end = 0.0
            timestamps = []
            while round(tmp_end, 3) < round(end, 3):
                tmp_end = start + sequence_len
                if round(tmp_end, 3) > round(float(self.duration), 3):
                    timestamps.append(
                        str(round(float(self.duration) - sequence_len, 3)) + "-" + str(round(float(self.duration), 3)))
                    return timestamps
                timestamps.append(str(round(start, 3)) + "-" + str(round(tmp_end, 3)))
                start += hop_size
            return timestamps

    def get_time_info(self, prediction_data):
        time_info = []
        first_el = True
        duration = float(list(prediction_data.keys())[-1].split("-")[1])
        for timestamp in prediction_data:
            time = timestamp.replace("-", " ").split(" ")
            time = [float(item) for item in time]
            pred = prediction_data.get(timestamp)[0]
            if first_el:
                first_el = False
            if str(pred) != str(get_dict_key_ignorecase(self.classes_label_idx, "noise")):
                if time_info:
                    if round(time_info[-1][0][1], 3) >= round(time[0], 3) and (str(time_info[-1][1]) == str(pred)):
                        time_info[-1] = (([round(time_info[-1][0][0], 3), round(time[1], 3)], pred))
                    elif round(time_info[-1][0][1], 3) >= round(time[0], 3) and (str(time_info[-1][1]) != str(pred)):
                        time_info.append(([time_info[-1][0][1], round(time[1], 3)], pred))
                    else:
                        time_info.append(([round(time[0], 3), round(time[1], 3)], pred))
                else:
                    time_info.append(([round(time[0], 3), round(time[1], 3)], pred))
        return time_info, duration

    def get_time_info_former(self, prediction_data):
        end = 0.0
        start = 0.0
        target = str(get_dict_key_ignorecase(self.classes_label_idx, "noise"))
        time_info = []
        first_el = True
        pred = None
        duration = float(list(prediction_data.keys())[-1].split("-")[1])
        for timestamp in prediction_data:
            time = timestamp.replace("-", " ").split(" ")
            prev_pred = pred
            pred = prediction_data.get(timestamp)[0]
            if first_el:
                prev_pred = pred
                first_el = False
            if (str(pred) != str(get_dict_key_ignorecase(self.classes_label_idx, "noise")) and (
            (str(pred) == str(prev_pred)))) or (str(prev_pred) == str(
                    get_dict_key_ignorecase(self.classes_label_idx, "noise"))):
                if target == str(get_dict_key_ignorecase(self.classes_label_idx, "noise")):
                    start = float(time[0])
                    end = float(time[1])
                    target = pred
                else:
                    end += float(self.hop)
            else:
                if str(target) != str(get_dict_key_ignorecase(self.classes_label_idx, "noise")):
                    if time_info:
                        if round(time_info[-1][0][1], 3) >= round(start, 3) and (str(time_info[-1][1]) == target):
                            time_info[-1] = (([round(time_info[-1][0][0], 3), round(end, 3)], target))
                        elif round(time_info[-1][0][1], 3) >= round(start, 3) and (str(time_info[-1][1]) != target):
                            time_info.append(([time_info[-1][0][1], round(end, 3)], target))
                        else:
                            time_info.append(([round(start, 3), round(end, 3)], target))

                        if str(pred) != str(get_dict_key_ignorecase(self.classes_label_idx, "noise")):
                            time_info.append(([time_info[-1][0][1], round(float(time[-1]), 3)], pred))

                    else:
                        time_info.append(([round(start, 3), round(end, 3)], target))
                    target = str(get_dict_key_ignorecase(self.classes_label_idx, "noise"))

        if str(target) != str(get_dict_key_ignorecase(self.classes_label_idx, "noise")):
            if time_info:
                if round(time_info[-1][0][1], 3) >= round(start, 3):
                    time_info[-1] = (([round(time_info[-1][0][0], 3), round(end, 3)], target))
                else:
                    time_info.append(([round(start, 3), round(end, 3)], target))
            else:
                time_info.append(([round(start, 3), round(end, 3)], target))
        return time_info, duration

    def get_time_info_orig(self, prediction_data):
        end = 0.0
        start = 0.0
        target = False
        time_info = []
        duration = float(list(prediction_data.keys())[-1].split("-")[1])
        for timestamp in prediction_data:
            time = timestamp.replace("-", " ").split(" ")
            pred = prediction_data.get(timestamp)[0]
            if pred:
                if not target:
                    start = float(time[0])
                    end = float(time[1])
                    target = True
                else:
                    end += float(self.hop)
            else:
                if target:
                    if time_info:
                        if round(time_info[-1][0][1], 3) >= round(start, 3):
                            time_info[-1] = (([round(time_info[-1][0][0], 3), round(end, 3)], 1))
                        else:
                            time_info.append(([round(start, 3), round(end, 3)], 1))
                    else:
                        time_info.append(([round(start, 3), round(end, 3)], 1))
                    target = False
        if target:
            if time_info:
                if round(time_info[-1][0][1], 3) >= round(start, 3):
                    time_info[-1] = (([round(time_info[-1][0][0], 3), round(end, 3)], 1))
                else:
                    time_info.append(([round(start, 3), round(end, 3)], 1))
            else:
                time_info.append(([round(start, 3), round(end, 3)], 1))
        return time_info, duration

    def add_noise_parts_to_time_info(self, time_info, duration):
        noise_time_info = []
        if not time_info:
            noise_time_info.append(([0.0, duration], get_dict_key_ignorecase(self.classes_label_idx, "noise")))
            return noise_time_info
        else:
            for index in range(len(time_info)):
                if index < len(time_info) - 1:
                    noise_time_info.append(time_info[index])
                    if time_info[index][0][1] != time_info[index + 1][0][0]:
                        noise_time_info.append(([time_info[index][0][1], time_info[index + 1][0][0]],
                                                get_dict_key_ignorecase(self.classes_label_idx, "noise")))
                else:
                    noise_time_info.append(time_info[index])
            if noise_time_info[0][0][0] != 0.0:
                noise_time_info.insert(0, (
                [0.0, noise_time_info[0][0][0]], get_dict_key_ignorecase(self.classes_label_idx, "noise")))
            if noise_time_info[-1][0][1] != duration:
                noise_time_info.append(
                    ([noise_time_info[-1][0][1], duration], get_dict_key_ignorecase(self.classes_label_idx, "noise")))
            return noise_time_info

    def get_overlapped_threshold_dependend_annotation_data(self, labeled_data, duration):
        threshold_depended_annotation_data_overlapped = []
        noise_time_info = self.add_noise_parts_to_time_info(labeled_data, duration)
        for index in range(len(noise_time_info)):
            threshold_depended_annotation_data_overlapped.append((self.classes_idx_label.get(
                int(noise_time_info[index][1])), -1, noise_time_info[index][0][0], noise_time_info[index][0][1], False))
        return threshold_depended_annotation_data_overlapped

    def write_annotation_file(self, path_annotation_file, annotation_data):
        iterator = 1
        annotated_prediction_file = open(path_annotation_file, 'w')
        annotated_prediction_file.write("\t".join(self.needed_annotation_columns) + "\n")
        for label, __, start, end, __ in annotation_data:
            if label == "noise" and self.config_data.get("noise_in_anno") == None:
                continue
            else:
                annotated_prediction_file.write(
                    str(iterator) + "\tSpectrogram_1\t1\t" + str(start) + "\t" + str(end) + "\t" + str(
                        self.data_opts.get("fmin")) + "\t" + str(self.data_opts.get("fmax")) + "\t" + label + "\t \n")
            iterator += 1
        annotated_prediction_file.close()

    def list_all_files_in_dir(self, path):
        onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        return onlyfiles

    def process(self):
        path = self.config_data.get("prediction_dir")
        prediction_files = self.list_all_files_in_dir(path)
        for pred_file in prediction_files:
            if pred_file.find("_predict_output") != -1:
                time_prob = self.read_prediction_file(path + "/" + pred_file)
                labeled_data, duration = self.get_all_network_prediction_data(time_prob,
                                                                              self.config_data.get("threshold"))
                threshold_depended_annotation_data_overlapped = self.get_overlapped_threshold_dependend_annotation_data(
                    labeled_data, duration)
                if platform.system() == "Windows":
                    self.write_annotation_file(
                        self.config_data["output_dir"] + "\\" + pred_file + ".annotation.result.txt",
                        threshold_depended_annotation_data_overlapped)
                else:
                    self.write_annotation_file(
                        self.config_data["output_dir"] + "/" + pred_file + ".annotation.result.txt",
                        threshold_depended_annotation_data_overlapped)
                self.logger.info("annotation file=" + str() + " has been successfully created!")


def startEvaluation(values):
    config_data = getEvalConfig(values)
    evaluator = setup_evaluator(config=config_data)
    evaluator.config_data = config_data
    evaluator.init_logger()
    evaluator.process()