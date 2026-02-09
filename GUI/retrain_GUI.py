import utils.PySimpleGUIQt as sg
import os
import sys
import logging
import multiprocessing as mp
import threading
from pathlib import Path

working_directory=os.getcwd()

sg.set_options(font=("Arial Bold",14))

re_src_dir_label=sg.Text("Enter ANIMAL-SPOT source directory:")
re_src_dir_input=sg.InputText(key="-re_src_dir-")
re_src_dir_filebrowser=sg.FolderBrowse(initial_folder=working_directory)

re_data_dir_label=sg.Text("Enter data directory:")
re_data_dir_input=sg.InputText(key="-re_data_dir-")
re_data_dir_filebrowser=sg.FolderBrowse(initial_folder=working_directory)

re_noise_dir_label=sg.Text("Enter noise directory:")
re_noise_dir_input=sg.InputText(key="-re_noise_dir-")
re_noise_dir_filebrowser=sg.FolderBrowse(initial_folder=working_directory)

re_cache_dir_label=sg.Text("Enter cache directory:")
re_cache_dir_input=sg.InputText(key="-re_cache_dir-")
re_cache_dir_filebrowser=sg.FolderBrowse(initial_folder=working_directory)

re_model_dir_label=sg.Text("Enter model file path:")
re_model_dir_input=sg.InputText(key="-re_model_dir-")
re_model_dir_filebrowser=sg.FileBrowse(initial_folder=working_directory)

re_checkpoint_dir_label=sg.Text("Enter checkpoint directory:")
re_checkpoint_dir_input=sg.InputText(key="-re_checkpoint_dir-")
re_checkpoint_dir_filebrowser=sg.FolderBrowse(initial_folder=working_directory)

re_log_dir_label=sg.Text("Enter log directory:")
re_log_dir_input=sg.InputText(key="-re_log_dir-")
re_log_dir_filebrowser=sg.FolderBrowse(initial_folder=working_directory)

re_summary_dir_label=sg.Text("Enter summary directory:")
re_summary_dir_input=sg.InputText(key="-re_summary_dir-")
re_summary_dir_filebrowser=sg.FolderBrowse(initial_folder=working_directory)

re_debug_label=sg.Text("Enable debug:")
re_debug_checkbox=sg.Checkbox(text="", default=False, key="-re_debug-")

re_start_from_scratch_label=sg.Text("Start from scratch:")
re_start_from_scratch_checkbox=sg.Checkbox(text="", default=True, key="-re_start_from_scratch-")

re_augmentation_label=sg.Text("Use augmentations:")
re_augmentation_checkbox=sg.Checkbox(text="", default=True, key="-re_augmentation-")

re_jit_save_label=sg.Text("Use jit save:")
re_jit_save_checkbox=sg.Checkbox(text="", default=False, key="-re_jit_save-")

re_no_cuda_label=sg.Text("Use cuda:")
re_no_cuda_checkbox=sg.Checkbox(text="", default=True, key="-re_no_cuda-")

re_filter_broken_audio_label=sg.Text("Filter broken audio:")
re_filter_broken_audio_checkbox=sg.Checkbox(text="", default=False, key="-re_filter_broken_audio-")

re_min_max_norm_label=sg.Text("Use min max normalization:")
re_min_max_norm_checkbox=sg.Checkbox(text="", default=True, key="-re_min_max_norm-")

re_sequence_len_label=sg.Text("Enter ANIMAL-SPOT window size in ms:")
re_sequence_len_input=sg.InputText(key="-re_sequence_len-", default_text="20")
re_sequence_len_reset=sg.Button(button_text="default", key="re_default_sequence_len")

re_max_train_epochs_label=sg.Text("Enter maximum number of epochs:")
re_max_train_epochs_input=sg.InputText(key="-re_max_train_epochs-", default_text="150")
re_max_train_epochs_reset=sg.Button(button_text="default", key="re_default_max_train_epochs")

re_epochs_per_eval_label=sg.Text("How many epochs per evaluation:")
re_epochs_per_eval_input=sg.InputText(key="-re_epochs_per_eval-", default_text="2")
re_epochs_per_eval_reset=sg.Button(button_text="default", key="re_default_epochs_per_eval")

re_batch_size_label=sg.Text("Batch size:")
re_batch_size_input=sg.InputText(key="-re_batch_size-", default_text="16")
re_batch_size_reset=sg.Button(button_text="default", key="re_default_batch_size")

re_num_workers_label=sg.Text("Number of worker:")
re_num_workers_input=sg.InputText(key="-re_num_workers-", default_text="0")
re_num_workers_reset=sg.Button(button_text="default", key="re_default_num_workers")

re_lr_label=sg.Text("Learning rate:")
re_lr_input=sg.InputText(key="-re_lr-", default_text="10e-5")
re_lr_reset=sg.Button(button_text="default", key="re_default_lr")

re_beta1_label=sg.Text("Adam optimizer beta1:")
re_beta1_input=sg.InputText(key="-re_beta1-", default_text="0.5")
re_beta1_reset=sg.Button(button_text="default", key="re_default_beta1")

re_lr_patience_epochs_label=sg.Text("Learning rate patience in epochs:")
re_lr_patience_epochs_input=sg.InputText(key="-re_lr_patience_epochs-", default_text="8")
re_lr_patience_epochs_reset=sg.Button(button_text="default", key="re_default_lr_patience_epochs")

re_lr_decay_factor_label=sg.Text("Learning rate decay:")
re_lr_decay_factor_input=sg.InputText(key="-re_lr_decay_factor-", default_text="0.5")
re_lr_decay_factor_reset=sg.Button(button_text="default", key="re_default_lr_decay_factor")

re_early_stopping_patience_epochs_label=sg.Text("Early stopping patience in epochs:")
re_early_stopping_patience_epochs_input=sg.InputText(key="-re_early_stopping_patience_epochs-", default_text="20")
re_early_stopping_patience_epochs_reset=sg.Button(button_text="default", key="re_default_early_stopping_patience_epochs")

re_freq_compression_label=sg.Text("Frequency compression method:")
re_freq_compression_input=sg.InputText(key="-re_freq_compression-", default_text="linear")
re_freq_compression_reset=sg.Button(button_text="default", key="re_default_freq_compression")

re_n_freq_bins_label=sg.Text("Number of frequency bins:")
re_n_freq_bins_input=sg.InputText(key="-re_n_freq_bins-", default_text="256")
re_n_freq_bins_reset=sg.Button(button_text="default", key="re_default_n_freq_bins")

re_n_fft_label=sg.Text("FFT window size:")
re_n_fft_input=sg.InputText(key="-re_n_fft-", default_text="256")
re_n_fft_reset=sg.Button(button_text="default", key="re_default_n_fft")

re_hop_length_label=sg.Text("FFT hop size:")
re_hop_length_input=sg.InputText(key="-re_hop_length-", default_text="128")
re_hop_length_reset=sg.Button(button_text="default", key="re_default_hop_length")

re_resnet_label=sg.Text("Resnet:")
re_resnet_input=sg.InputText(key="-re_resnet-", default_text="18")
re_resnet_reset=sg.Button(button_text="default", key="re_default_resnet")

re_conv_kernel_size_label=sg.Text("Convolutional kernel size:")
re_conv_kernel_size_input=sg.InputText(key="-re_conv_kernel_size-", default_text="7")
re_conv_kernel_size_reset=sg.Button(button_text="default", key="re_default_conv_kernel_size")

re_num_classes_label=sg.Text("Number of classes:")
re_num_classes_input=sg.InputText(key="-re_num_classes-", default_text="2")
re_num_classes_reset=sg.Button(button_text="default", key="re_default_num_classes")

re_max_pool_label=sg.Text("Max pooling:")
re_max_pool_input=sg.InputText(key="-re_max_pool-", default_text="2")
re_max_pool_reset=sg.Button(button_text="default", key="re_default_max_pool")

re_fmin_label=sg.Text("Frequency minimum:")
re_fmin_input=sg.InputText(key="-re_fmin-", default_text="18000")
re_fmin_reset=sg.Button(button_text="default", key="re_default_fmin")

re_fmax_label=sg.Text("Frequency minimum:")
re_fmax_input=sg.InputText(key="-re_fmax-", default_text="90000")
re_fmax_reset=sg.Button(button_text="default", key="re_default_fmax")

re_save_config_button=sg.FileSaveAs(button_text="save settings")
re_save_config_Input=sg.Input(key="re_save_config", enable_events=True, visible=False)
re_load_config_button=sg.FileBrowse(button_text="load settings")
re_load_config_Input=sg.Input(key="re_load_config", enable_events=True, visible=False)

re_start_prediction_button=sg.Button(button_text="Start Training", key="re_start")
#t_output = sg.Output(size=(67, 10))

#sg.Print('Re-routing train_GUI to Debug stdout', do_not_reroute_stdout=False)

Retrain_layout=[
    [re_model_dir_label,re_model_dir_input,re_model_dir_filebrowser],
    [re_src_dir_label,re_src_dir_input,re_src_dir_filebrowser],
    [re_data_dir_label,re_data_dir_input,re_data_dir_filebrowser],
    [re_noise_dir_label,re_noise_dir_input,re_noise_dir_filebrowser],
    [re_cache_dir_label,re_cache_dir_input,re_cache_dir_filebrowser],
    [re_checkpoint_dir_label,re_checkpoint_dir_input,re_checkpoint_dir_filebrowser],
    [re_log_dir_label,re_log_dir_input,re_log_dir_filebrowser],
    [re_summary_dir_label,re_summary_dir_input,re_summary_dir_filebrowser],
    [re_debug_label,re_debug_checkbox],
    [re_start_from_scratch_label,re_start_from_scratch_checkbox],
    [re_augmentation_label,re_augmentation_checkbox],
    [re_min_max_norm_label,re_min_max_norm_checkbox],
    [re_jit_save_label,re_jit_save_checkbox],
    [re_no_cuda_label,re_no_cuda_checkbox],
    [re_filter_broken_audio_label,re_filter_broken_audio_checkbox],
    [re_sequence_len_label,re_sequence_len_input,re_sequence_len_reset],
    [re_max_train_epochs_label,re_max_train_epochs_input,re_max_train_epochs_reset],
    [re_epochs_per_eval_label,re_epochs_per_eval_input,re_epochs_per_eval_reset],
    [re_batch_size_label,re_batch_size_input,re_batch_size_reset],
    [re_num_workers_label,re_num_workers_input,re_num_workers_reset],
    [re_lr_label,re_lr_input,re_lr_reset],
    [re_beta1_label,re_beta1_input,re_beta1_reset],
    [re_lr_patience_epochs_label,re_lr_patience_epochs_input,re_lr_patience_epochs_reset],
    [re_lr_decay_factor_label,re_lr_decay_factor_input,re_lr_decay_factor_reset],
    [re_early_stopping_patience_epochs_label,re_early_stopping_patience_epochs_input,re_early_stopping_patience_epochs_reset],
    [re_freq_compression_label,re_freq_compression_input,re_freq_compression_reset],
    [re_n_freq_bins_label,re_n_freq_bins_input,re_n_freq_bins_reset],
    [re_n_fft_label,re_n_fft_input,re_n_fft_reset],
    [re_hop_length_label,re_hop_length_input,re_hop_length_reset],
    [re_resnet_label,re_resnet_input,re_resnet_reset],
    [re_conv_kernel_size_label,re_conv_kernel_size_input,re_conv_kernel_size_reset],
    [re_num_classes_label,re_num_classes_input,re_num_classes_reset],
    [re_max_pool_label,re_max_pool_input,re_max_pool_reset],
    [re_fmin_label,re_fmin_input,re_fmin_reset],
    [re_fmax_label,re_fmax_input,re_fmax_reset],
    [re_save_config_button, re_save_config_Input],
    [re_load_config_button, re_load_config_Input],
    [re_start_prediction_button],
    #[t_output]
]
retrain_column = [[sg.Column(Retrain_layout, scrollable=True, size=(1000,700))]]

def getRetrainGUI():
    return retrain_column

def RetrainhandleInput(event, values, window):
    if event == "re_default_sequence_len":
        window['-re_sequence_len-'].update("20")
        values['-re_sequence_len-'] = "20"
    if event == "re_default_max_train_epochs":
        window['-re_max_train_epochs-'].update("150")
        values['-re_max_train_epochs-'] = "150"
    if event == "re_default_epochs_per_eval":
        window['-re_epochs_per_eval-'].update("2")
        values['-re_epochs_per_eval-'] = "2"
    if event == "re_default_batch_size":
        window['-re_batch_size-'].update("16")
        values['-re_batch_size-'] = "16"
    if event == "re_default_num_workers":
        window['-re_num_workers-'].update("0")
        values['-re_num_workers-'] = "0"
    if event == "re_default_lr":
        window['-re_lr-'].update("10e-5")
        values['-re_lr-'] = "10e-5"
    if event == "re_default_beta1":
        window['-re_beta1-'].update("0.5")
        values['-re_beta1-'] = "0.5"
    if event == "re_default_lr_patience_epochs":
        window['-re_lr_patience_epochs-'].update("8")
        values['-re_lr_patience_epochs-'] = "8"
    if event == "re_default_lr_decay_factor":
        window['-re_lr_decay_factor-'].update("0.5")
        values['-re_lr_decay_factor-'] = "0.5"
    if event == "re_default_early_stopping_patience_epochs":
        window['-re_early_stopping_patience_epochs-'].update("20")
        values['-re_early_stopping_patience_epochs-'] = "20"
    if event == "re_default_freq_compression":
        window['-re_freq_compression-'].update("linear")
        values['-re_freq_compression-'] = "linear"
    if event == "re_default_n_freq_bins":
        window['-re_n_freq_bins-'].update("256")
        values['-re_n_freq_bins-'] = "256"
    if event == "re_default_n_fft":
        window['-re_n_fft-'].update("256")
        values['-re_n_fft-'] = "256"
    if event == "re_default_hop_length":
        window['-re_hop_length-'].update("128")
        values['-re_hop_length-'] = "128"
    if event == "re_default_resnet":
        window['-re_resnet-'].update("18")
        values['-re_resnet-'] = "18"
    if event == "re_default_conv_kernel_size":
        window['-re_conv_kernel_size-'].update("7")
        values['-re_conv_kernel_size-'] = "7"
    if event == "re_default_num_classes":
        window['-re_num_classes-'].update("2")
        values['-re_num_classes-'] = "2"
    if event == "re_default_max_pool":
        window['-re_max_pool-'].update("2")
        values['-re_max_pool-'] = "2"
    if event == "re_default_fmin":
        window['-re_fmin-'].update("18000")
        values['-re_fmin-'] = "18000"
    if event == "re_default_fmax":
        window['-re_fmax-'].update("90000")
        values['-re_fmax-'] = "90000"
    if event == "re_save_config":
        generateTrainConfig(values=values)
    if event == "re_load_config":
        loadTrainConfig(values=values, window=window)
    if event == "re_start":
        startTraining(values=values)

def generateTrainConfig(values):
    file = open(re_save_config_Input.get(), "w")
    # Directorys
    if values["-re_src_dir-"] == "":
        sg.popup_error("ANIMAL-SPOT source File not set")
        file.close()
        return
    file.write("src_dir=" + str(values["-re_src_dir-"]) + "/\n")

    if values["-re_model_dir-"] == "":
        sg.popup_error("Model directory not specified")
        file.close()
        return
    file.write("model_dir=" + str(values["-re_model_dir-"]) + "/\n")

    if values["-re_data_dir-"] == "":
        sg.popup_error("Data directory not specified")
        file.close()
        return
    file.write("data_dir=" + str(values["-re_data_dir-"]) + "/\n")

    if values["-re_checkpoint_dir-"] == "":
        sg.popup_error("Checkpoint directory not specified")
        file.close()
        return
    file.write("checkpoint_dir=" + str(values["-re_checkpoint_dir-"]) + "/\n")

    if values["-re_summary_dir-"] == "":
        sg.popup_error("Summary directory not specified")
        file.close()
        return
    file.write("summary_dir=" + str(values["-re_summary_dir-"]) + "/\n")

    # optional Directories
    if values["-re_noise_dir-"] != "":  # optional
        file.write("noise_dir=" + str(values["-re_noise_dir-"]) + "/\n")

    if values["-re_cache_dir-"] != "":  # optional
        file.write("cache_dir=" + str(values["-re_cache_dir-"]) + "/\n")

    if values["-re_log_dir-"] != "":  # optional
        file.write("log_dir=" + str(values["-re_log_dir-"]) + "/\n")

    # Boolean Parameter
    if values["-re_debug-"] is True:  # optional
        file.write("debug=" + str(values["-re_debug-"]) + "\n")
    else:
        file.write("debug=False" + "\n")

    if values["-re_start_from_scratch-"] is True:  # optional
        file.write("start_from_scratch=" + str(values["-re_start_from_scratch-"]) + "\n")
    else:
        file.write("start_from_scratch=False" + "\n")

    if values["-re_no_cuda-"] == False:  # optional
        file.write("no_cuda=True" + "\n")
    else:
        file.write("no_cuda=False" + "\n")

    if values["-re_augmentation-"] is True:  # optional
        file.write("augmentation=" + str(values["-re_augmentation-"]) + "\n")
    else:
        file.write("augmentation=False" + "\n")

    if values["-re_jit_save-"] is True:  # optional
        file.write("jit_save=" + str(values["-re_jit_save-"]) + "\n")
    else:
        file.write("jit_save=False" + "\n")

    if values["-re_filter_broken_audio-"] is True:  # optional
        file.write("filter_broken_audio=" + str(values["-re_filter_broken_audio-"]) + "\n")
    else:
        file.write("filter_broken_audio=False" + "\n")

    if values["-re_min_max_norm-"] is True:  # optional
        file.write("min_max_norm=" + str(values["-re_min_max_norm-"]) + "\n")
    else:
        file.write("min_max_norm=False" + "\n")

    # Number Parameter
    if values["-re_sequence_len-"] != "":
        file.write("sequence_len=" + str(values["-re_sequence_len-"]) + "\n")

    if values["-re_max_train_epochs-"] != "":
        file.write("max_train_epochs=" + str(values["-re_max_train_epochs-"]) + "\n")

    if values["-re_epochs_per_eval-"] != "":
        file.write("epochs_per_eval=" + str(values["-re_epochs_per_eval-"]) + "\n")

    if values["-re_batch_size-"] != "":
        file.write("batch_size=" + str(values["-re_batch_size-"]) + "\n")

    if values["-re_num_workers-"] != "":
        file.write("num_workers=" + str(values["-re_num_workers-"]) + "\n")

    if values["-re_lr-"] != "":
        file.write("lr=" + str(values["-re_lr-"]) + "\n")

    if values["-re_beta1-"] != "":
        file.write("beta1=" + str(values["-re_beta1-"]) + "\n")

    if values["-re_lr_patience_epochs-"] != "":
        file.write("lr_patience_epochs=" + str(values["-re_lr_patience_epochs-"]) + "\n")

    if values["-re_lr_decay_factor-"] != "":
        file.write("lr_decay_factor=" + str(values["-re_lr_decay_factor-"]) + "\n")

    if values["-re_early_stopping_patience_epochs-"] != "":
        file.write("early_stopping_patience_epochs=" + str(values["-re_early_stopping_patience_epochs-"]) + "\n")

    if values["-re_freq_compression-"] != "":
        file.write("freq_compression=" + str(values["-re_freq_compression-"]) + "\n")

    if values["-re_n_freq_bins-"] != "":
        file.write("n_freq_bins=" + str(values["-re_n_freq_bins-"]) + "\n")

    if values["-re_n_fft-"] != "":
        file.write("n_fft=" + str(values["-re_n_fft-"]) + "\n")

    if values["-re_hop_length-"] != "":
        file.write("hop_length=" + str(values["-re_hop_length-"]) + "\n")

    if values["-re_resnet-"] != "":
        file.write("resnet=" + str(values["-re_resnet-"]) + "\n")

    if values["-re_conv_kernel_size-"] != "":
        file.write("conv_kernel_size=" + str(values["-re_conv_kernel_size-"]) + "\n")

    if values["-re_num_classes-"] != "":
        file.write("num_classes=" + str(values["-re_num_classes-"]) + "\n")

    if values["-re_max_pool-"] != "":
        file.write("max_pool=" + str(values["-re_max_pool-"]) + "\n")

    if values["-re_fmin-"] != "":
        file.write("fmin=" + str(values["-re_fmin-"]) + "\n")

    if values["-re_fmax-"] != "":
        file.write("fmax=" + str(values["-re_fmax-"]) + "\n")

    file.close()

def loadTrainConfig(values, window):
    file = open(re_load_config_Input.get())
    lines = file.readlines()
    for line in lines:
        if line.__contains__("#") or line.__contains__("*"):
            continue
        if line.__contains__("src_dir="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_src_dir-'].update(val)
            values['-re_src_dir-'] = val
        if line.__contains__("data_dir="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_data_dir-'].update(val)
            values['-re_data_dir-'] = val
        if line.__contains__("noise_dir="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_noise_dir-'].update(val)
            values['-re_noise_dir-'] = val
        if line.__contains__("cache_dir="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_cache_dir-'].update(val)
            values['-re_cache_dir-'] = val
        if line.__contains__("model_dir="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_model_dir-'].update(val)
            values['-re_model_dir-'] = val
        if line.__contains__("checkpoint_dir="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_checkpoint_dir-'].update(val)
            values['-re_checkpoint_dir-'] = val
        if line.__contains__("log_dir="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_log_dir-'].update(val)
            values['-re_log_dir-'] = val
        if line.__contains__("summary_dir="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_summary_dir-'].update(val)
            values['-re_summary_dir-'] = val

        if line.__contains__("debug="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-re_debug-'].update(True)
                values['-re_debug-'] = True
            else:
                window['-re_debug-'].update(False)
                values['-re_debug-'] = False
        if line.__contains__("start_from_scratch="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-re_start_from_scratch-'].update(True)
                values['-re_start_from_scratch-'] = True
            else:
                window['-re_start_from_scratch-'].update(False)
                values['-re_start_from_scratch-'] = False
        if line.__contains__("augmentation="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-re_augmentation-'].update(True)
                values['-re_augmentation-'] = True
            else:
                window['-re_augmentation-'].update(False)
                values['-re_augmentation-'] = False
        if line.__contains__("jit_save="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-re_jit_save-'].update(True)
                values['-re_jit_save-'] = True
            else:
                window['-re_jit_save-'].update(False)
                values['-re_jit_save-'] = False
        if line.__contains__("no_cuda="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-re_no_cuda-'].update(False)
                values['-re_no_cuda-'] = False
            else:
                window['-re_no_cuda-'].update(True)
                values['-re_no_cuda-'] = True
        if line.__contains__("filter_broken_audio="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-re_filter_broken_audio-'].update(True)
                values['-re_filter_broken_audio-'] = True
            else:
                window['-re_filter_broken_audio-'].update(False)
                values['-re_filter_broken_audio-'] = False
        if line.__contains__("min_max_norm="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-re_min_max_norm-'].update(True)
                values['-re_min_max_norm-'] = True
            else:
                window['-re_min_max_norm-'].update(False)
                values['-re_min_max_norm-'] = False

        if line.__contains__("sequence_len="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_sequence_len-'].update(val)
            values['-re_sequence_len-'] = val
        if line.__contains__("max_train_epochs="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_max_train_epochs-'].update(val)
            values['-re_max_train_epochs-'] = val
        if line.__contains__("epochs_per_eval="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_epochs_per_eval-'].update(val)
            values['-re_epochs_per_eval-'] = val
        if line.__contains__("batch_size="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_batch_size-'].update(val)
            values['-re_batch_size-'] = val
        if line.__contains__("num_workers="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_num_workers-'].update(val)
            values['-re_num_workers-'] = val
        if line.__contains__("lr="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_lr-'].update(val)
            values['-re_lr-'] = val
        if line.__contains__("beta1="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_beta1-'].update(val)
            values['-re_beta1-'] = val
        if line.__contains__("lr_patience_epochs="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_lr_patience_epochs-'].update(val)
            values['-re_lr_patience_epochs-'] = val
        if line.__contains__("lr_decay_factor="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_lr_decay_factor-'].update(val)
            values['-re_lr_decay_factor-'] = val
        if line.__contains__("early_stopping_patience_epochs="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_early_stopping_patience_epochs-'].update(val)
            values['-re_early_stopping_patience_epochs-'] = val
        if line.__contains__("freq_compression="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_freq_compression-'].update(val)
            values['-re_freq_compression-'] = val
        if line.__contains__("n_freq_bins="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_n_freq_bins-'].update(val)
            values['-re_n_freq_bins-'] = val
        if line.__contains__("n_fft="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_n_fft-'].update(val)
            values['-re_n_fft-'] = val
        if line.__contains__("hop_length="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_hop_length-'].update(val)
            values['-re_hop_length-'] = val
        if line.__contains__("resnet="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_resnet-'].update(val)
            values['-re_resnet-'] = val
        if line.__contains__("conv_kernel_size="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_conv_kernel_size-'].update(val)
            values['-re_conv_kernel_size-'] = val
        if line.__contains__("num_classes="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_num_classes-'].update(val)
            values['-re_num_classes-'] = val
        if line.__contains__("max_pool="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_max_pool-'].update(val)
            values['-re_max_pool-'] = val
        if line.__contains__("fmin="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_fmin-'].update(val)
            values['-re_fmin-'] = val
        if line.__contains__("fmax="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-re_fmax-'].update(val)
            values['-re_fmax-'] = val
    file.close()

def startTraining(values):
    GUI_path = Path(os.path.abspath(os.path.dirname(__file__)))
    ASG_path = GUI_path.parent.absolute()
    pythonexe = 'python'  # os.path.join(ASG_path, 'venv/Scripts/python.exe')
    train_cmd = pythonexe + " -W ignore::UserWarning"
    if values["-re_src_dir-"] == "":
        sg.popup_error("ANIMAL-SPOT source File not set")
        return
    elif not os.path.isfile(values["-re_src_dir-"]+"/main.py"):
        sg.popup_error("Source File error")
        return
    train_cmd = train_cmd + " " + values["-re_src_dir-"]+"/main.py"

    #Directorys
    if values["-re_model_dir-"] == "":
        sg.popup_error("Model directory not specified")
        return
    train_cmd = train_cmd + " --model_dir " + values["-re_model_dir-"]+"/"

    if values["-re_data_dir-"] == "":
        sg.popup_error("Data directory not specified")
        return
    train_cmd = train_cmd + " --data_dir " + values["-re_data_dir-"]+"/"

    if values["-re_checkpoint_dir-"] == "":
        sg.popup_error("Checkpoint directory not specified")
        return
    train_cmd = train_cmd + " --checkpoint_dir " + values["-re_checkpoint_dir-"]+"/"

    if values["-re_summary_dir-"] == "":
        sg.popup_error("Summary directory not specified")
        return
    train_cmd = train_cmd + " --summary_dir " + values["-re_summary_dir-"]+"/"

    #optional Directories
    if values["-re_noise_dir-"] != "": # optional
        train_cmd = train_cmd + " --noise_dir " + values["-re_noise_dir-"]+"/"

    if values["-re_cache_dir-"] != "":  # optional
        train_cmd = train_cmd + " --cache_dir " + values["-re_cache_dir-"]+"/"

    if values["-re_log_dir-"] != "":  # optional
        train_cmd = train_cmd + " --log_dir " + values["-re_log_dir-"]+"/"

    #Boolean Parameter
    if values["-re_debug-"] is True:  # optional
        train_cmd = train_cmd + " --debug"

    if values["-re_start_from_scratch-"] is True:  # optional
        train_cmd = train_cmd + " --start_from_scratch"

    if values["-re_no_cuda-"] == False:  # optional
        train_cmd = train_cmd + " --no_cuda"

    if values["-re_augmentation-"] is True:  # optional
        train_cmd = train_cmd + " --augmentation"

    if values["-re_jit_save-"] is True:  # optional
        train_cmd = train_cmd + " --jit_save"

    if values["-re_filter_broken_audio-"] is True:  # optional
        train_cmd = train_cmd + " --filter_broken_audio"

    if values["-re_min_max_norm-"] is True:  # optional
        train_cmd = train_cmd + " --min_max_norm"

    # Number Parameter
    if values["-re_sequence_len-"] != "":
        train_cmd = train_cmd + " --sequence_len " + values["-re_sequence_len-"]

    if values["-re_max_train_epochs-"] != "":
        train_cmd = train_cmd + " --max_train_epochs " + values["-re_max_train_epochs-"]

    if values["-re_epochs_per_eval-"] != "":
        train_cmd = train_cmd + " --epochs_per_eval " + values["-re_epochs_per_eval-"]

    if values["-re_batch_size-"] != "":
        train_cmd = train_cmd + " --batch_size " + values["-re_batch_size-"]

    if values["-re_num_workers-"] != "":
        train_cmd = train_cmd + " --num_workers " + values["-re_num_workers-"]

    if values["-re_lr-"] != "":
        train_cmd = train_cmd + " --lr " + values["-re_lr-"]

    if values["-re_beta1-"] != "":
        train_cmd = train_cmd + " --beta1 " + values["-re_beta1-"]

    if values["-re_lr_patience_epochs-"] != "":
        train_cmd = train_cmd + " --lr_patience_epochs " + values["-re_lr_patience_epochs-"]

    if values["-re_lr_decay_factor-"] != "":
        train_cmd = train_cmd + " --lr_decay_factor " + values["-re_lr_decay_factor-"]

    if values["-re_early_stopping_patience_epochs-"] != "":
        train_cmd = train_cmd + " --early_stopping_patience_epochs " + values["-re_early_stopping_patience_epochs-"]

    if values["-re_freq_compression-"] != "":
        train_cmd = train_cmd + " --freq_compression " + values["-re_freq_compression-"]

    if values["-re_n_freq_bins-"] != "":
        train_cmd = train_cmd + " --n_freq_bins " + values["-re_n_freq_bins-"]

    if values["-re_n_fft-"] != "":
        train_cmd = train_cmd + " --n_fft " + values["-re_n_fft-"]

    if values["-re_hop_length-"] != "":
        train_cmd = train_cmd + " --hop_length " + values["-re_hop_length-"]

    if values["-re_resnet-"] != "":
        train_cmd = train_cmd + " --resnet " + values["-re_resnet-"]

    if values["-re_conv_kernel_size-"] != "":
        train_cmd = train_cmd + " --conv_kernel_size " + values["-re_conv_kernel_size-"]

    if values["-re_num_classes-"] != "":
        train_cmd = train_cmd + " --num_classes " + values["-re_num_classes-"]

    if values["-re_max_pool-"] != "":
        train_cmd = train_cmd + " --max_pool " + values["-re_max_pool-"]

    if values["-re_fmin-"] != "":
        train_cmd = train_cmd + " --fmin " + values["-re_fmin-"]

    if values["-re_fmax-"] != "":
        train_cmd = train_cmd + " --fmax " + values["-re_fmax-"]

    t1 = threading.Thread(target=startTrainCommand, args=(train_cmd,))
    t1.start()
    sg.popup('The Training has started in Thread ' + str(t1.ident) +
             '\nPlease look at the console to follow the training progress or errors. '
             'Also please do not start another Thread unless you know what you are doing!')

def startTrainCommand(train_cmd):
    logger = logging.getLogger('training animal-spot')
    stream_handler = logging.StreamHandler()
    logger.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.info("Training Command: " + train_cmd)
    logger.info("Start Training!!!")
    os.system(train_cmd)

"""def spawnDaemon(train_cmd):
    # fork the first time (to make a non-session-leader child process)
    try:
        pid = os.fork() #does not work on Windows
    except Exception:
        pass
    if pid != 0:
        # parent (calling) process is all done
        return

    # detach from controlling terminal (to make child a session-leader)
    os.setsid()
    try:
        pid = os.fork()
    except Exception:
        pass
    if pid != 0:
        # child process is all done
        os._exit(0)

    # grandchild process now non-session-leader, detached from parent
    # grandchild process must now close all open files
    try:
        maxfd = os.sysconf("SC_OPEN_MAX")
    except (AttributeError, ValueError):
        maxfd = 1024

    for fd in range(maxfd):
        try:
            os.close(fd)
        except OSError:  # ERROR, fd wasn't open to begin with (ignored)
            pass

    # redirect stdin, stdout and stderr to /dev/null
    os.open(os.devnull, os.O_RDWR)  # standard input (0)
    os.dup2(0, 1)
    os.dup2(0, 2)

    # and finally let's execute the executable for the daemon!
    try:
        os.system(train_cmd)
    except Exception:
        os._exit(255)
"""