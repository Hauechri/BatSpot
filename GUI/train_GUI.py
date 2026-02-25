import utils.PySimpleGUIQt as sg
import os
import sys
import logging
import multiprocessing as mp
import threading
from pathlib import Path

GUI_path = Path(os.path.abspath(os.path.dirname(__file__)))
ASG_path = GUI_path.parent.absolute()
sys.path.insert(0, str(ASG_path))

from animal_spot.main import start_train, build_args

working_directory=os.getcwd()

sg.set_options(font=("Arial Bold",14))

#t_src_dir_label=sg.Text("ANIMAL-SPOT source directory:")
#t_src_dir_input=sg.InputText(key="-t_src_dir-")
#t_src_dir_filebrowser=sg.FolderBrowse(initial_folder=working_directory)

t_data_dir_label=sg.Text("Path folder training examples:")
t_data_dir_input=sg.InputText(key="-t_data_dir-")
t_data_dir_filebrowser=sg.FolderBrowse(initial_folder=working_directory)

t_noise_dir_label=sg.Text("Path folder augmentation noise examples (optional):")
t_noise_dir_input=sg.InputText(key="-t_noise_dir-")
t_noise_dir_filebrowser=sg.FolderBrowse(initial_folder=working_directory)

t_cache_dir_label=sg.Text("Path folder to store cache (optional):")
t_cache_dir_input=sg.InputText(key="-t_cache_dir-")
t_cache_dir_filebrowser=sg.FolderBrowse(initial_folder=working_directory)

#t_train_or_retrain_label=sg.Text("Enable retrain:")
#t_train_or_retrain=sg.Checkbox(text="", default=False, key="-t_tore-")

t_model_dir_label=sg.Text("Path folder to store model:")
t_model_dir_input=sg.InputText(key="-t_model_dir-")
t_model_dir_folderbrowser=sg.FolderBrowse(initial_folder=working_directory)

#t_model_dir_relabel=sg.Text("Path to model file for retraining:")
#t_model_dir_reinput=sg.InputText(key="-t_model_redir-")
#t_model_dir_refilebrowser=sg.FileBrowse(initial_folder=working_directory)

t_checkpoint_dir_label=sg.Text("Path folder to store checkpoints:")
t_checkpoint_dir_input=sg.InputText(key="-t_checkpoint_dir-")
t_checkpoint_dir_filebrowser=sg.FolderBrowse(initial_folder=working_directory)

t_log_dir_label=sg.Text("Path folder to store log:")
t_log_dir_input=sg.InputText(key="-t_log_dir-")
t_log_dir_filebrowser=sg.FolderBrowse(initial_folder=working_directory)

t_summary_dir_label=sg.Text("Path folder to store summary:")
t_summary_dir_input=sg.InputText(key="-t_summary_dir-")
t_summary_dir_filebrowser=sg.FolderBrowse(initial_folder=working_directory)

t_transfer_learning_label=sg.Text("Use retraining or transfer learning:")
t_transfer_learning=sg.Checkbox(text="", default=False, key="-t_transfer-")

t_transfer_model_label=sg.Text("Path model for retraining or transfer learning:")
t_transfer_model_input=sg.InputText(key="-t_transfer_model-")
t_transfer_model_filebrowser=sg.FileBrowse(initial_folder=working_directory)

t_debug_label=sg.Text("Enable debug:")
t_debug_checkbox=sg.Checkbox(text="", default=False, key="-t_debug-")

t_start_from_scratch_label=sg.Text("Start from scratch:")
t_start_from_scratch_checkbox=sg.Checkbox(text="", default=True, key="-t_start_from_scratch-")

t_augmentation_label=sg.Text("Use augmentations:")
t_augmentation_checkbox=sg.Checkbox(text="", default=True, key="-t_augmentation-")

t_jit_save_label=sg.Text("Use jit save:")
t_jit_save_checkbox=sg.Checkbox(text="", default=False, key="-t_jit_save-")

t_no_cuda_label=sg.Text("Use cuda:")
t_no_cuda_checkbox=sg.Checkbox(text="", default=True, key="-t_no_cuda-")

t_filter_broken_audio_label=sg.Text("Filter broken audio:")
t_filter_broken_audio_checkbox=sg.Checkbox(text="", default=False, key="-t_filter_broken_audio-")

t_min_max_norm_label=sg.Text("Use min max normalization:")
t_min_max_norm_checkbox=sg.Checkbox(text="", default=True, key="-t_min_max_norm-")

t_samplingrate_label=sg.Text("Samplingrate in Hz:")
t_samplingrate_input=sg.InputText(key="-t_samplingrate-", default_text="250000")
t_samplingrate_reset=sg.Button(button_text="default", key="t_default_samplingrate")

t_sequence_len_label=sg.Text("Window size in ms:")
t_sequence_len_input=sg.InputText(key="-t_sequence_len-", default_text="20")
t_sequence_len_reset=sg.Button(button_text="default", key="t_default_sequence_len")

t_max_train_epochs_label=sg.Text("Maximum number of epochs:")
t_max_train_epochs_input=sg.InputText(key="-t_max_train_epochs-", default_text="150")
t_max_train_epochs_reset=sg.Button(button_text="default", key="t_default_max_train_epochs")

t_epochs_per_eval_label=sg.Text("How many epochs per evaluation:")
t_epochs_per_eval_input=sg.InputText(key="-t_epochs_per_eval-", default_text="2")
t_epochs_per_eval_reset=sg.Button(button_text="default", key="t_default_epochs_per_eval")

t_batch_size_label=sg.Text("Batch size:")
t_batch_size_input=sg.InputText(key="-t_batch_size-", default_text="16")
t_batch_size_reset=sg.Button(button_text="default", key="t_default_batch_size")

t_num_workers_label=sg.Text("Number of worker:")
t_num_workers_input=sg.InputText(key="-t_num_workers-", default_text="0")
t_num_workers_reset=sg.Button(button_text="default", key="t_default_num_workers")

t_lr_label=sg.Text("Learning rate:")
t_lr_input=sg.InputText(key="-t_lr-", default_text="10e-5")
t_lr_reset=sg.Button(button_text="default", key="t_default_lr")

t_beta1_label=sg.Text("Adam optimizer beta1:")
t_beta1_input=sg.InputText(key="-t_beta1-", default_text="0.5")
t_beta1_reset=sg.Button(button_text="default", key="t_default_beta1")

t_lr_patience_epochs_label=sg.Text("Learning rate patience in epochs:")
t_lr_patience_epochs_input=sg.InputText(key="-t_lr_patience_epochs-", default_text="8")
t_lr_patience_epochs_reset=sg.Button(button_text="default", key="t_default_lr_patience_epochs")

t_lr_decay_factor_label=sg.Text("Learning rate decay:")
t_lr_decay_factor_input=sg.InputText(key="-t_lr_decay_factor-", default_text="0.5")
t_lr_decay_factor_reset=sg.Button(button_text="default", key="t_default_lr_decay_factor")

t_early_stopping_patience_epochs_label=sg.Text("Early stopping patience in epochs:")
t_early_stopping_patience_epochs_input=sg.InputText(key="-t_early_stopping_patience_epochs-", default_text="20")
t_early_stopping_patience_epochs_reset=sg.Button(button_text="default", key="t_default_early_stopping_patience_epochs")

t_freq_compression_label=sg.Text("Frequency compression method:")
t_freq_compression_input=sg.InputText(key="-t_freq_compression-", default_text="linear")
t_freq_compression_reset=sg.Button(button_text="default", key="t_default_freq_compression")

t_n_freq_bins_label=sg.Text("Number of frequency bins:")
t_n_freq_bins_input=sg.InputText(key="-t_n_freq_bins-", default_text="256")
t_n_freq_bins_reset=sg.Button(button_text="default", key="t_default_n_freq_bins")

t_n_fft_label=sg.Text("FFT window size:")
t_n_fft_input=sg.InputText(key="-t_n_fft-", default_text="256")
t_n_fft_reset=sg.Button(button_text="default", key="t_default_n_fft")

t_hop_length_label=sg.Text("FFT hop size:")
t_hop_length_input=sg.InputText(key="-t_hop_length-", default_text="128")
t_hop_length_reset=sg.Button(button_text="default", key="t_default_hop_length")

t_resnet_label=sg.Text("Resnet:")
t_resnet_input=sg.InputText(key="-t_resnet-", default_text="18")
t_resnet_reset=sg.Button(button_text="default", key="t_default_resnet")

t_conv_kernel_size_label=sg.Text("Convolutional kernel size:")
t_conv_kernel_size_input=sg.InputText(key="-t_conv_kernel_size-", default_text="7")
t_conv_kernel_size_reset=sg.Button(button_text="default", key="t_default_conv_kernel_size")

t_num_classes_label=sg.Text("Number of classes:")
t_num_classes_input=sg.InputText(key="-t_num_classes-", default_text="2")
t_num_classes_reset=sg.Button(button_text="default", key="t_default_num_classes")

t_max_pool_label=sg.Text("Max pooling:")
t_max_pool_input=sg.InputText(key="-t_max_pool-", default_text="2")
t_max_pool_reset=sg.Button(button_text="default", key="t_default_max_pool")

t_fmin_label=sg.Text("Frequency minimum:")
t_fmin_input=sg.InputText(key="-t_fmin-", default_text="18000")
t_fmin_reset=sg.Button(button_text="default", key="t_default_fmin")

t_fmax_label=sg.Text("Frequency maximum:")
t_fmax_input=sg.InputText(key="-t_fmax-", default_text="90000")
t_fmax_reset=sg.Button(button_text="default", key="t_default_fmax")

t_save_config_button=sg.FileSaveAs(button_text="Save settings")
t_save_config_Input=sg.Input(key="t_save_config", enable_events=True, visible=False)
t_load_config_button=sg.FileBrowse(button_text="Load settings")
t_load_config_Input=sg.Input(key="t_load_config", enable_events=True, visible=False)

t_start_prediction_button=sg.Button(button_text="Start training", key="t_start")
#t_output = sg.Output(size=(67, 10))

#sg.Print('Re-routing train_GUI to Debug stdout', do_not_reroute_stdout=False)

train_layout=[
    #[t_train_or_retrain_label, t_train_or_retrain],
    #[t_model_dir_relabel, t_model_dir_reinput, t_model_dir_refilebrowser],
    #[t_src_dir_label,t_src_dir_input,t_src_dir_filebrowser],
    [t_data_dir_label,t_data_dir_input,t_data_dir_filebrowser],
    [t_noise_dir_label,t_noise_dir_input,t_noise_dir_filebrowser],
    [t_model_dir_label, t_model_dir_input, t_model_dir_folderbrowser],
    [t_checkpoint_dir_label,t_checkpoint_dir_input,t_checkpoint_dir_filebrowser],
    [t_log_dir_label,t_log_dir_input,t_log_dir_filebrowser],
    [t_summary_dir_label,t_summary_dir_input,t_summary_dir_filebrowser],
    [t_cache_dir_label,t_cache_dir_input,t_cache_dir_filebrowser],
    [t_transfer_learning_label, t_transfer_learning],
    [t_transfer_model_label,t_transfer_model_input,t_transfer_model_filebrowser],
    [t_debug_label,t_debug_checkbox],
    [t_start_from_scratch_label,t_start_from_scratch_checkbox],
    [t_augmentation_label,t_augmentation_checkbox],
    [t_filter_broken_audio_label,t_filter_broken_audio_checkbox],
    [t_samplingrate_label,t_samplingrate_input,t_samplingrate_reset],
    [t_sequence_len_label,t_sequence_len_input,t_sequence_len_reset],
    [t_max_train_epochs_label,t_max_train_epochs_input,t_max_train_epochs_reset],
    [t_early_stopping_patience_epochs_label,t_early_stopping_patience_epochs_input,t_early_stopping_patience_epochs_reset],
    [t_n_freq_bins_label,t_n_freq_bins_input,t_n_freq_bins_reset],
    [t_n_fft_label,t_n_fft_input,t_n_fft_reset],
    [t_fmin_label,t_fmin_input,t_fmin_reset],
    [t_fmax_label,t_fmax_input,t_fmax_reset],
    [t_num_classes_label,t_num_classes_input,t_num_classes_reset],
    [t_epochs_per_eval_label,t_epochs_per_eval_input,t_epochs_per_eval_reset],
    [t_min_max_norm_label,t_min_max_norm_checkbox],
    [t_jit_save_label,t_jit_save_checkbox],
    [t_no_cuda_label,t_no_cuda_checkbox],
    [t_batch_size_label,t_batch_size_input,t_batch_size_reset],
    [t_num_workers_label,t_num_workers_input,t_num_workers_reset],
    [t_lr_label,t_lr_input,t_lr_reset],
    [t_beta1_label,t_beta1_input,t_beta1_reset],
    [t_lr_patience_epochs_label,t_lr_patience_epochs_input,t_lr_patience_epochs_reset],
    [t_lr_decay_factor_label,t_lr_decay_factor_input,t_lr_decay_factor_reset],
    [t_freq_compression_label,t_freq_compression_input,t_freq_compression_reset],
    [t_hop_length_label,t_hop_length_input,t_hop_length_reset],
    [t_resnet_label,t_resnet_input,t_resnet_reset],
    [t_conv_kernel_size_label,t_conv_kernel_size_input,t_conv_kernel_size_reset],
    [t_max_pool_label,t_max_pool_input,t_max_pool_reset],
    [t_save_config_button, t_save_config_Input],
    [t_load_config_button, t_load_config_Input],
    [t_start_prediction_button],
    #[t_output]
]
train_column = [[sg.Column(train_layout, scrollable=True, size=(1000,700))]]

def getTrainGUI():
    return train_column

def TrainhandleInput(event, values, window):
    if event == "t_default_samplingrate":
        window['-t_samplingrate-'].update("250000")
        values['-t_samplingrate-'] = "250000"
    if event == "t_default_sequence_len":
        window['-t_sequence_len-'].update("20")
        values['-t_sequence_len-'] = "20"
    if event == "t_default_max_train_epochs":
        window['-t_max_train_epochs-'].update("150")
        values['-t_max_train_epochs-'] = "150"
    if event == "t_default_epochs_per_eval":
        window['-t_epochs_per_eval-'].update("2")
        values['-t_epochs_per_eval-'] = "2"
    if event == "t_default_batch_size":
        window['-t_batch_size-'].update("16")
        values['-t_batch_size-'] = "16"
    if event == "t_default_num_workers":
        window['-t_num_workers-'].update("0")
        values['-t_num_workers-'] = "0"
    if event == "t_default_lr":
        window['-t_lr-'].update("10e-5")
        values['-t_lr-'] = "10e-5"
    if event == "t_default_beta1":
        window['-t_beta1-'].update("0.5")
        values['-t_beta1-'] = "0.5"
    if event == "t_default_lr_patience_epochs":
        window['-t_lr_patience_epochs-'].update("8")
        values['-t_lr_patience_epochs-'] = "8"
    if event == "t_default_lr_decay_factor":
        window['-t_lr_decay_factor-'].update("0.5")
        values['-t_lr_decay_factor-'] = "0.5"
    if event == "t_default_early_stopping_patience_epochs":
        window['-t_early_stopping_patience_epochs-'].update("20")
        values['-t_early_stopping_patience_epochs-'] = "20"
    if event == "t_default_freq_compression":
        window['-t_freq_compression-'].update("linear")
        values['-t_freq_compression-'] = "linear"
    if event == "t_default_n_freq_bins":
        window['-t_n_freq_bins-'].update("256")
        values['-t_n_freq_bins-'] = "256"
    if event == "t_default_n_fft":
        window['-t_n_fft-'].update("256")
        values['-t_n_fft-'] = "256"
    if event == "t_default_hop_length":
        window['-t_hop_length-'].update("128")
        values['-t_hop_length-'] = "128"
    if event == "t_default_resnet":
        window['-t_resnet-'].update("18")
        values['-t_resnet-'] = "18"
    if event == "t_default_conv_kernel_size":
        window['-t_conv_kernel_size-'].update("7")
        values['-t_conv_kernel_size-'] = "7"
    if event == "t_default_num_classes":
        window['-t_num_classes-'].update("2")
        values['-t_num_classes-'] = "2"
    if event == "t_default_max_pool":
        window['-t_max_pool-'].update("2")
        values['-t_max_pool-'] = "2"
    if event == "t_default_fmin":
        window['-t_fmin-'].update("18000")
        values['-t_fmin-'] = "18000"
    if event == "t_default_fmax":
        window['-t_fmax-'].update("90000")
        values['-t_fmax-'] = "90000"
    if event == "t_save_config":
        generateTrainConfig(values=values)
    if event == "t_load_config":
        loadTrainConfig(values=values, window=window)
    if event == "t_start":
        startTraining(values=values)

def generateTrainConfig(values):
    file = open(t_save_config_Input.get(), "w")
    # Directorys
    #if values["-t_src_dir-"] == "":
    #    sg.popup_error("ANIMAL-SPOT source File not set")
    #    file.close()
    #    return
    #file.write("src_dir=" + str(values["-t_src_dir-"]) + "\n")

    if values["-t_model_dir-"] == "":
        sg.popup_error("Model save directory not specified")
        file.close()
        return
    file.write("model_dir=" + str(values["-t_model_dir-"]) + "\n")

    if values["-t_data_dir-"] == "":
        sg.popup_error("Data directory not specified")
        file.close()
        return
    file.write("data_dir=" + str(values["-t_data_dir-"]) + "\n")

    if values["-t_checkpoint_dir-"] == "":
        sg.popup_error("Checkpoint directory not specified")
        file.close()
        return
    file.write("checkpoint_dir=" + str(values["-t_checkpoint_dir-"]) + "\n")

    if values["-t_summary_dir-"] == "":
        sg.popup_error("Summary directory not specified")
        file.close()
        return
    file.write("summary_dir=" + str(values["-t_summary_dir-"]) + "\n")

    # optional Directories
    if values["-t_noise_dir-"] != "":  # optional
        file.write("noise_dir=" + str(values["-t_noise_dir-"]) + "\n")

    if values["-t_cache_dir-"] != "":  # optional
        file.write("cache_dir=" + str(values["-t_cache_dir-"]) + "\n")

    if values["-t_log_dir-"] != "":  # optional
        file.write("log_dir=" + str(values["-t_log_dir-"]) + "\n")

    if values["-t_transfer_model-"] != "":  # optional
        file.write("transfer_model=" + str(values["-t_transfer_model-"]) + "\n")

    # Boolean Parameter
    if values["-t_debug-"] is True:  # optional
        file.write("debug=" + str(values["-t_debug-"]) + "\n")
    else:
        file.write("debug=False" + "\n")

    if values["-t_start_from_scratch-"] is True:  # optional
        file.write("start_from_scratch=" + str(values["-t_start_from_scratch-"]) + "\n")
    else:
        file.write("start_from_scratch=False" + "\n")

    if values["-t_transfer-"] == False:  # optional
        file.write("transfer=True" + "\n")
    else:
        file.write("transfer=False" + "\n")

    if values["-t_no_cuda-"] == False:  # optional
        file.write("no_cuda=True" + "\n")
    else:
        file.write("no_cuda=False" + "\n")


    if values["-t_augmentation-"] is True:  # optional
        file.write("augmentation=" + str(values["-t_augmentation-"]) + "\n")
    else:
        file.write("augmentation=False" + "\n")

    if values["-t_jit_save-"] is True:  # optional
        file.write("jit_save=" + str(values["-t_jit_save-"]) + "\n")
    else:
        file.write("jit_save=False" + "\n")

    if values["-t_filter_broken_audio-"] is True:  # optional
        file.write("filter_broken_audio=" + str(values["-t_filter_broken_audio-"]) + "\n")
    else:
        file.write("filter_broken_audio=False" + "\n")

    if values["-t_min_max_norm-"] is True:  # optional
        file.write("min_max_norm=" + str(values["-t_min_max_norm-"]) + "\n")
    else:
        file.write("min_max_norm=False" + "\n")

    # Number Parameter
    if values["-t_samplingrate-"] != "":
        file.write("sr=" + str(values["-t_samplingrate-"]) + "\n")

    if values["-t_sequence_len-"] != "":
        file.write("sequence_len=" + str(values["-t_sequence_len-"]) + "\n")

    if values["-t_max_train_epochs-"] != "":
        file.write("max_train_epochs=" + str(values["-t_max_train_epochs-"]) + "\n")

    if values["-t_epochs_per_eval-"] != "":
        file.write("epochs_per_eval=" + str(values["-t_epochs_per_eval-"]) + "\n")

    if values["-t_batch_size-"] != "":
        file.write("batch_size=" + str(values["-t_batch_size-"]) + "\n")

    if values["-t_num_workers-"] != "":
        file.write("num_workers=" + str(values["-t_num_workers-"]) + "\n")

    if values["-t_lr-"] != "":
        file.write("lr=" + str(values["-t_lr-"]) + "\n")

    if values["-t_beta1-"] != "":
        file.write("beta1=" + str(values["-t_beta1-"]) + "\n")

    if values["-t_lr_patience_epochs-"] != "":
        file.write("lr_patience_epochs=" + str(values["-t_lr_patience_epochs-"]) + "\n")

    if values["-t_lr_decay_factor-"] != "":
        file.write("lr_decay_factor=" + str(values["-t_lr_decay_factor-"]) + "\n")

    if values["-t_early_stopping_patience_epochs-"] != "":
        file.write("early_stopping_patience_epochs=" + str(values["-t_early_stopping_patience_epochs-"]) + "\n")

    if values["-t_freq_compression-"] != "":
        file.write("freq_compression=" + str(values["-t_freq_compression-"]) + "\n")

    if values["-t_n_freq_bins-"] != "":
        file.write("n_freq_bins=" + str(values["-t_n_freq_bins-"]) + "\n")

    if values["-t_n_fft-"] != "":
        file.write("n_fft=" + str(values["-t_n_fft-"]) + "\n")

    if values["-t_hop_length-"] != "":
        file.write("hop_length=" + str(values["-t_hop_length-"]) + "\n")

    if values["-t_resnet-"] != "":
        file.write("resnet=" + str(values["-t_resnet-"]) + "\n")

    if values["-t_conv_kernel_size-"] != "":
        file.write("conv_kernel_size=" + str(values["-t_conv_kernel_size-"]) + "\n")

    if values["-t_num_classes-"] != "":
        file.write("num_classes=" + str(values["-t_num_classes-"]) + "\n")

    if values["-t_max_pool-"] != "":
        file.write("max_pool=" + str(values["-t_max_pool-"]) + "\n")

    if values["-t_fmin-"] != "":
        file.write("fmin=" + str(values["-t_fmin-"]) + "\n")

    if values["-t_fmax-"] != "":
        file.write("fmax=" + str(values["-t_fmax-"]) + "\n")

    file.close()

def loadTrainConfig(values, window):
    file = open(t_load_config_Input.get())
    lines = file.readlines()
    for line in lines:
        if line.__contains__("#") or line.__contains__("*"):
            continue
        #if line.__contains__("src_dir="):
        #    val = line.split("=")[1]
        #    val = val.split("\n")[0]
        #    window['-t_src_dir-'].update(val)
        #    values['-t_src_dir-'] = val
        if line.__contains__("data_dir="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_data_dir-'].update(val)
            values['-t_data_dir-'] = val
        if line.__contains__("noise_dir="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_noise_dir-'].update(val)
            values['-t_noise_dir-'] = val
        if line.__contains__("cache_dir="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_cache_dir-'].update(val)
            values['-t_cache_dir-'] = val
        if line.__contains__("model_dir="):
                val = line.split("=")[1]
                val = val.split("\n")[0]
                window['-t_model_dir-'].update(val)
                values['-t_model_dir-'] = val

        if line.__contains__("checkpoint_dir="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_checkpoint_dir-'].update(val)
            values['-t_checkpoint_dir-'] = val
        if line.__contains__("transfer_model="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_transfer_model-'].update(val)
            values['-t_transfer_model-'] = val
        if line.__contains__("log_dir="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_log_dir-'].update(val)
            values['-t_log_dir-'] = val
        if line.__contains__("summary_dir="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_summary_dir-'].update(val)
            values['-t_summary_dir-'] = val

        if line.__contains__("debug="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-t_debug-'].update(True)
                values['-t_debug-'] = True
            else:
                window['-t_debug-'].update(False)
                values['-t_debug-'] = False
        if line.__contains__("start_from_scratch="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-t_start_from_scratch-'].update(True)
                values['-t_start_from_scratch-'] = True
            else:
                window['-t_start_from_scratch-'].update(False)
                values['-t_start_from_scratch-'] = False
        if line.__contains__("transfer="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-t_transfer-'].update(True)
                values['-t_transfer-'] = True
            else:
                window['-t_transfer-'].update(False)
                values['-t_transfer-'] = False
        if line.__contains__("augmentation="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-t_augmentation-'].update(True)
                values['-t_augmentation-'] = True
            else:
                window['-t_augmentation-'].update(False)
                values['-t_augmentation-'] = False
        if line.__contains__("jit_save="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-t_jit_save-'].update(True)
                values['-t_jit_save-'] = True
            else:
                window['-t_jit_save-'].update(False)
                values['-t_jit_save-'] = False
        if line.__contains__("no_cuda="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-t_no_cuda-'].update(False)
                values['-t_no_cuda-'] = False
            else:
                window['-t_no_cuda-'].update(True)
                values['-t_no_cuda-'] = True
        if line.__contains__("filter_broken_audio="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-t_filter_broken_audio-'].update(True)
                values['-t_filter_broken_audio-'] = True
            else:
                window['-t_filter_broken_audio-'].update(False)
                values['-t_filter_broken_audio-'] = False
        if line.__contains__("min_max_norm="):
            val = line.split("=")[1]
            if val.__contains__("True") or val.__contains__("true"):
                window['-t_min_max_norm-'].update(True)
                values['-t_min_max_norm-'] = True
            else:
                window['-t_min_max_norm-'].update(False)
                values['-t_min_max_norm-'] = False

        if line.__contains__("sr="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_samplingrate-'].update(val)
            values['-t_samplingrate-'] = val
        if line.__contains__("sequence_len="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_sequence_len-'].update(val)
            values['-t_sequence_len-'] = val
        if line.__contains__("max_train_epochs="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_max_train_epochs-'].update(val)
            values['-t_max_train_epochs-'] = val
        if line.__contains__("epochs_per_eval="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_epochs_per_eval-'].update(val)
            values['-t_epochs_per_eval-'] = val
        if line.__contains__("batch_size="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_batch_size-'].update(val)
            values['-t_batch_size-'] = val
        if line.__contains__("num_workers="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_num_workers-'].update(val)
            values['-t_num_workers-'] = val
        if line.__contains__("lr="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_lr-'].update(val)
            values['-t_lr-'] = val
        if line.__contains__("beta1="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_beta1-'].update(val)
            values['-t_beta1-'] = val
        if line.__contains__("lr_patience_epochs="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_lr_patience_epochs-'].update(val)
            values['-t_lr_patience_epochs-'] = val
        if line.__contains__("lr_decay_factor="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_lr_decay_factor-'].update(val)
            values['-t_lr_decay_factor-'] = val
        if line.__contains__("early_stopping_patience_epochs="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_early_stopping_patience_epochs-'].update(val)
            values['-t_early_stopping_patience_epochs-'] = val
        if line.__contains__("freq_compression="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_freq_compression-'].update(val)
            values['-t_freq_compression-'] = val
        if line.__contains__("n_freq_bins="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_n_freq_bins-'].update(val)
            values['-t_n_freq_bins-'] = val
        if line.__contains__("n_fft="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_n_fft-'].update(val)
            values['-t_n_fft-'] = val
        if line.__contains__("hop_length="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_hop_length-'].update(val)
            values['-t_hop_length-'] = val
        if line.__contains__("resnet="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_resnet-'].update(val)
            values['-t_resnet-'] = val
        if line.__contains__("conv_kernel_size="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_conv_kernel_size-'].update(val)
            values['-t_conv_kernel_size-'] = val
        if line.__contains__("num_classes="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_num_classes-'].update(val)
            values['-t_num_classes-'] = val
        if line.__contains__("max_pool="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_max_pool-'].update(val)
            values['-t_max_pool-'] = val
        if line.__contains__("fmin="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_fmin-'].update(val)
            values['-t_fmin-'] = val
        if line.__contains__("fmax="):
            val = line.split("=")[1]
            val = val.split("\n")[0]
            window['-t_fmax-'].update(val)
            values['-t_fmax-'] = val
    file.close()

def startTraining_old(values):
    GUI_path = Path(os.path.abspath(os.path.dirname(__file__)))
    ASG_path = GUI_path.parent.absolute()
    pythonexe = 'python'#os.path.join(ASG_path, 'venv/Scripts/python.exe')
    train_cmd = pythonexe + " -W ignore::UserWarning"
    #if values["-t_src_dir-"] == "":
    #    sg.popup_error("ANIMAL-SPOT source File not set")
    #    return
    #elif not os.path.isfile(values["-t_src_dir-"]+"/main.py"):
    #    sg.popup_error("Source File error")
    #    return
    #train_cmd = train_cmd + " " + values["-t_src_dir-"]+"/main.py"

    #Directorys
    if values["-t_model_dir-"] == "":
        sg.popup_error("Model directory not specified")
        return
    train_cmd = train_cmd + " --model_dir " + values["-t_model_dir-"]

    if values["-t_data_dir-"] == "":
        sg.popup_error("Data directory not specified")
        return
    train_cmd = train_cmd + " --data_dir " + values["-t_data_dir-"]

    if values["-t_checkpoint_dir-"] == "":
        sg.popup_error("Checkpoint directory not specified")
        return
    train_cmd = train_cmd + " --checkpoint_dir " + values["-t_checkpoint_dir-"]

    if values["-t_summary_dir-"] == "":
        sg.popup_error("Summary directory not specified")
        return
    train_cmd = train_cmd + " --summary_dir " + values["-t_summary_dir-"]

    #optional Directories
    if values["-t_noise_dir-"] != "": # optional
        train_cmd = train_cmd + " --noise_dir " + values["-t_noise_dir-"]

    if values["-t_cache_dir-"] != "":  # optional
        train_cmd = train_cmd + " --cache_dir " + values["-t_cache_dir-"]

    if values["-t_log_dir-"] != "":  # optional
        train_cmd = train_cmd + " --log_dir " + values["-t_log_dir-"]

    if values["-t_transfer_model-"] != "":  # optional
        train_cmd = train_cmd + " --transfer_dir " + values["-t_transfer_model-"]

    #Boolean Parameter
    if values["-t_debug-"] is True:  # optional
        train_cmd = train_cmd + " --debug"

    if values["-t_start_from_scratch-"] is True:  # optional
        train_cmd = train_cmd + " --start_from_scratch"

    if values["-t_transfer-"] is True:  # optional
        train_cmd = train_cmd + " --transfer"

    if values["-t_no_cuda-"] is False:  # optional # use Cuda:?
        train_cmd = train_cmd + " --no_cuda"

    if values["-t_augmentation-"] is True:  # optional
        train_cmd = train_cmd + " --augmentation"

    if values["-t_jit_save-"] is True:  # optional
        train_cmd = train_cmd + " --jit_save"

    if values["-t_filter_broken_audio-"] is True:  # optional
        train_cmd = train_cmd + " --filter_broken_audio"

    if values["-t_min_max_norm-"] is True:  # optional
        train_cmd = train_cmd + " --min_max_norm"

    # Number Parameter
    if values["-t_samplingrate-"] != "":
        train_cmd = train_cmd + " --sr " + values["-t_samplingrate-"]

    if values["-t_sequence_len-"] != "":
        train_cmd = train_cmd + " --sequence_len " + values["-t_sequence_len-"]

    if values["-t_max_train_epochs-"] != "":
        train_cmd = train_cmd + " --max_train_epochs " + values["-t_max_train_epochs-"]

    if values["-t_epochs_per_eval-"] != "":
        train_cmd = train_cmd + " --epochs_per_eval " + values["-t_epochs_per_eval-"]

    if values["-t_batch_size-"] != "":
        train_cmd = train_cmd + " --batch_size " + values["-t_batch_size-"]

    if values["-t_num_workers-"] != "":
        train_cmd = train_cmd + " --num_workers " + values["-t_num_workers-"]

    if values["-t_lr-"] != "":
        train_cmd = train_cmd + " --lr " + values["-t_lr-"]

    if values["-t_beta1-"] != "":
        train_cmd = train_cmd + " --beta1 " + values["-t_beta1-"]

    if values["-t_lr_patience_epochs-"] != "":
        train_cmd = train_cmd + " --lr_patience_epochs " + values["-t_lr_patience_epochs-"]

    if values["-t_lr_decay_factor-"] != "":
        train_cmd = train_cmd + " --lr_decay_factor " + values["-t_lr_decay_factor-"]

    if values["-t_early_stopping_patience_epochs-"] != "":
        train_cmd = train_cmd + " --early_stopping_patience_epochs " + values["-t_early_stopping_patience_epochs-"]

    if values["-t_freq_compression-"] != "":
        train_cmd = train_cmd + " --freq_compression " + values["-t_freq_compression-"]

    if values["-t_n_freq_bins-"] != "":
        train_cmd = train_cmd + " --n_freq_bins " + values["-t_n_freq_bins-"]

    if values["-t_n_fft-"] != "":
        train_cmd = train_cmd + " --n_fft " + values["-t_n_fft-"]

    if values["-t_hop_length-"] != "":
        train_cmd = train_cmd + " --hop_length " + values["-t_hop_length-"]

    if values["-t_resnet-"] != "":
        train_cmd = train_cmd + " --resnet " + values["-t_resnet-"]

    if values["-t_conv_kernel_size-"] != "":
        train_cmd = train_cmd + " --conv_kernel_size " + values["-t_conv_kernel_size-"]

    if values["-t_num_classes-"] != "":
        train_cmd = train_cmd + " --num_classes " + values["-t_num_classes-"]

    if values["-t_max_pool-"] != "":
        train_cmd = train_cmd + " --max_pool " + values["-t_max_pool-"]

    if values["-t_fmin-"] != "":
        train_cmd = train_cmd + " --fmin " + values["-t_fmin-"]

    if values["-t_fmax-"] != "":
        train_cmd = train_cmd + " --fmax " + values["-t_fmax-"]

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




def gui_values_to_arglist(values):
    args = []

    def add(flag, key):
        if values[key] != "":
            args.extend([flag, str(values[key])])

    def add_bool(flag, key):
        if values[key] is True:
            args.append(flag)

    # required
    add("--model_dir", "-t_model_dir-")
    add("--data_dir", "-t_data_dir-")
    add("--checkpoint_dir", "-t_checkpoint_dir-")
    add("--summary_dir", "-t_summary_dir-")

    # optional dirs
    add("--noise_dir", "-t_noise_dir-")
    add("--cache_dir", "-t_cache_dir-")
    add("--log_dir", "-t_log_dir-")
    add("--transfer_dir", "-t_transfer_model-")

    # booleans
    add_bool("--debug", "-t_debug-")
    add_bool("--start_from_scratch", "-t_start_from_scratch-")
    add_bool("--transfer", "-t_transfer-")
    add_bool("--augmentation", "-t_augmentation-")
    add_bool("--jit_save", "-t_jit_save-")
    add_bool("--filter_broken_audio", "-t_filter_broken_audio-")
    add_bool("--min_max_norm", "-t_min_max_norm-")

    # CUDA flag is inverted in your GUI
    if values["-t_no_cuda-"] is False:
        args.append("--no_cuda")

    # numeric
    add("--sr", "-t_samplingrate-")
    add("--sequence_len", "-t_sequence_len-")
    add("--max_train_epochs", "-t_max_train_epochs-")
    add("--epochs_per_eval", "-t_epochs_per_eval-")
    add("--batch_size", "-t_batch_size-")
    add("--num_workers", "-t_num_workers-")
    add("--lr", "-t_lr-")
    add("--beta1", "-t_beta1-")
    add("--lr_patience_epochs", "-t_lr_patience_epochs-")
    add("--lr_decay_factor", "-t_lr_decay_factor-")
    add("--early_stopping_patience_epochs", "-t_early_stopping_patience_epochs-")
    add("--freq_compression", "-t_freq_compression-")
    add("--n_freq_bins", "-t_n_freq_bins-")
    add("--n_fft", "-t_n_fft-")
    add("--hop_length", "-t_hop_length-")
    add("--resnet", "-t_resnet-")
    add("--conv_kernel_size", "-t_conv_kernel_size-")
    add("--num_classes", "-t_num_classes-")
    add("--max_pool", "-t_max_pool-")
    add("--fmin", "-t_fmin-")
    add("--fmax", "-t_fmax-")

    return args

def startTraining(values):
    arg_list = gui_values_to_arglist(values)

    def run():
        ARGS = build_args(arg_list)
        start_train(ARGS)

    t1 = threading.Thread(target=run, daemon=True)
    t1.start()

    sg.popup(
        'Training started in thread ' + str(t1.ident) +
        '\nCheck the console for progress.'
    )

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