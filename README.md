Repository for BatSpot - a CNN tool to detect and classify bat vocalisations.

# Content

- [Requirements](#requirement)
- [Installation](#installation)
- [Training](#training)
- [Retraining](#retraining)
- [Transfer learning](#transfer-learning)
- [Prediction](#prediction)
- [Translation](#translation)

# Requirements

For the GUI: none (if using the Windows exe) or Python 3.10 (if installing
in any other way).

For the underlying ANIMAL-SPOT source code (which can still run without the
GUI): Python >=3.8 & <=3.12.

# Installation

## Windows

### Using the exe

1. Download this zip file: **link**.

2. Unzip.

3. Click on the BatSpot.exe.

### Installing yourself

0. Make sure you have the correct version of Python (version 3.10.11). To test 
which version you have, simply open a Windows PowerShell and type  
`python -- version`. To download another version, go to this link: 
<https://www.python.org/downloads/> and make sure it is set as the default. 
Also, make sure to tick `Add python to PATH` during installation.

1. Download this repository and unzip.

2. Open a Command Prompt window (not PowerShell), make a virtual environment 
and install all requirements. To do this, copy the full path to the repository 
and update the first line, then run:

  ```
  cd /path/to/BatSpot-main
  python -m venv venv
  .\venv\Scripts\Activate
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt
  ```
  
Note that this installs torch for CPU only. If you want to use a GPU, see the
lines for Linux.

3. To start the GUI, copy the full path to the repository and update the first 
line, then run:

  ```
  cd /path/to/BatSpot-main
  .\venv\Scripts\Activate
  python GUI/start_GUI_tabs.py
  ```

4. The GUI should now open. If not, contact <simeonqs@hotmail.com> or open an 
issue on GitHub.

## Mac

Installation on Mac is still in the developmental phase and remains untested, 
please contact <simeonqs@hotmail.com> with any questions. 

0. Make sure you have the correct version of Python (version 3.10.11). To test 
which version you have, simply open a Terminal window and type 
`python -- version`. To change your Python version you first need Homebrew:

  ```
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```
  
  Then install the correct version of Python:
  
  ```
  brew install pyenv
  pyenv install 3.10.11
  pyenv global 3.10.11
  ```

1. Download this repository and unzip.

2. Make a virtual environment and install all requirements. Copy the full path
to the repository and update the first line, then run:

  ```
  cd /path/to/BatSpot-main
  python -m venv venv
  source venv/bin/activate
  pip install --upgrade pip setuptools wheel
  pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
  pip install -r requirements.txt
  ```

3. To start the GUI, copy the full path to the repository and update the first 
line, then run:

  ```
  cd /path/to/BatSpot-main
  source venv/bin/activate
  python GUI/start_GUI_tabs.py
  ```

4. The GUI should now open. If not, contact <simeonqs@hotmail.com> or open an 
issue on GitHub.

## Linux

0. Make sure you have the correct version of Python (version 3.10.11). To test 
which version you have, simply open a Terminal window and type 
`python -- version`. To change your Python version on Ubuntu/Debian:

  ```
  sudo apt update
  sudo apt install software-properties-common -y
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt update
  
  sudo apt install python3.10 python3.10-venv python3.10-dev -y
  
  python3.10 --version
  ```
  
  To make it your default run (otherwise it will still run the correct version
  in the virtual environment - venv):
  
  ```
  sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
  sudo update-alternatives --config python3
  ```

1. Download this repository and unzip.

2. Make a virtual environment and install all requirements. Copy the full path
to the repository and update the first line, then run:

  ```
  cd /path/to/BatSpot-main
  python3.10 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip setuptools wheel
  pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
      torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
  pip install -r requirements.txt
  ```

3. To start the GUI, copy the full path to the repository and update the first 
line, then run:

  ```
  cd /path/to/BatSpot-main
  source venv/bin/activate
  python GUI/start_GUI_tabs.py
  ```

4. The GUI should now open. If not, contact <simeonqs@hotmail.com> or open an 
issue on GitHub.

# Training

## Data preparation

BatSpot is trained on small audio clips with `.wav` extension (lower-case). 
Clips that are longer than the `Window size in ms` set in the GUI, will 
automatically be shortened by randomly selecting a section of the correct 
duration. Clips that are shorter will automatically be zero-buffered. The audio
files have to be named correctly for BatSpot to be able to retrieve all the 
meta data needed for training. This structure is the same as for the original
ANIMAL-SPOT (<https://github.com/ChristianBergler/ANIMAL-SPOT>) and the 
following is taken from that repository:

---------

`Filename Template: CLASSNAME-LABELINFO_ID_YEAR_TAPENAME_STARTTIMEMS_ENDTIMEMS.wav`

The entire file name consists of 6 elements, separated via the "\_" symbol. 
Consequently this type of symbol is not allowed within the strings, because it 
is considered as delimiter. Moreover, the "-" symbol has an important and 
special meaning. For separation of the two file name parts, *CLASSNAME* and 
*LABELINFO* (see file name template), the "-" symbol is mandatory and acts as 
delimiter in order to identify the *CLASSNAME*. Within all other parts of the 
file name template (e.g. *ID*, *YEAR*, *TAPENAME*) the "-" symbol can be used 
as element to concatenate several strings, as any other symbol, except "\_".

**1st-Element: CLASSNAME-LABELINFO** = The first part *CLASSNAME* has to be 
the name of the respective class, whereas the second part *LABELINFO* is 
optional, and could be used to provide additional label information, e.g. 
"orca-podA4callN7", followed by "_ID_YEAR...". If LABELINFO is not used, 
it is still important to keep the "-" symbol after *CLASSNAME*, as the first 
occurrence of "-" acts as delimiter. *CLASSNAME* ends afterthe first 
occurrence of "-".

**2nd-Element: ID** = unique *ID* (natural number) to identify the audio clip

**3rd-Element: YEAR** = the year the tape has been recorded

**4th-Element: TAPENAME** = name of the recorded tape. This is important for 
a proper split of the data into training, validation, and test. This is 
achieved by using the *TAPENAME* and *YEAR* as joint unique identifier, in 
order to avoid that samples of the same recording (year and tape) are spread 
over the distributions. It is therefore important to include many excerpts 
from different tapes, in order to ensure a proper and automatic data split. 

**5th-Element: STARTTIMEMS** = start time of the audio clip in milliseconds 
within the original recording (natural number)

**6th-Element: ENDTIMEMS** = end time of the audio clip in milliseconds 
within the original recording (natural number)

**Examples of valid filenames:**

*call-Orca-A12_929_2019_Rec-031-2018-10-19-06-59-59-ASWMUX231648_2949326_2949919*

CLASSNAME = call, LABELINFO = Orca-A12, ID = 929, YEAR = 2019, TAPENAME = 
Rec-031-2018-10-19-06-59-59-ASWMUX231648, STARTTIMEMS = 2949326, ENDTIMEMS = 
2949919

*noise-_2381_2010_101BC_149817_150055.wav*

CLASSNAME = noise, LABELINFO = , ID = 2381, YEAR = 2010, TAPENAME = 101BC, 
STARTTIMEMS = 149817, ENDTIMEMS = 150055

---------

It might seem daunting to create all these examples with the correct naming 
structure. Therefore we have created an R-script that can take the raw audio
recordings and selection tables from Raven LIte as input, and create the audio 
clips as output. Several examples of this script can be found here: 
<https://github.com/simeonqs/BatSpot_article/tree/main/analysis/code>.

## Train a new model

To train a new model, you can use the training tab.

1. Store all the audio examples (target, noise and/or all classes) in one 
folder. If needed, store all augmentation examples in another folder without 
sub folders (see below).

2. Start the GUI. (See installation guide above.)

3. Fill out the required fields, or load a previous config file using the 
`Load settings` button.

  - `Path folder training examples`:path to the folder with the audio examples 
  (.wav)
  - `Path folder augmentation noise examples (optional)`: the path to the 
  folder containing the noise examples for augmentation. These files are used 
  as artificial background noise by mixing it into the training examples. 
  This is optional. It leads to better generalisation, but could potentially 
  lead to worse performance, if noise is too loud or stereotyped.
  - `Path folder to store model`: path to the folder where the final model file
  will be stored. The file will be named `ANIMAL-SPOT.dk`.
  - `Path folder to store checkpoints`: path to the folder where checkpoints
  should be stored. These can be used to restart training, if it for some 
  reason stops. Simply press `Start training` again and make sure 
  `Start from scratch` is not selected.
  - `Path folder to store log`: path to the folder where the training log 
  should stored. This file contains the console output during the training 
  phase and can be useful to see how parameters where set, which data split 
  was used and how the model performed during validation and testing. The log 
  file will be named `TRAIN.log`.
  - `Path folder to store summary`: path to the folder where the summary files
  should stored. These are used to generate visualisation of the training, 
  validation and testing performance.
  - `Path folder to store cache (optional)`: path to the folder where cache 
  files should be stored. Only supply this if you think you need cache.
  - `Use retraining or transfer learning`: if checked the model will use a 
  previous model or checkpoint files from a previous training session to start 
  retraining or transfer learning. To start from a previous model file, supply 
  the path to the file in the next argument on the list. To start from a 
  checkpoint, make sure it is stored in the folder that you entered above for 
  the checkpoints. Transfer learning (where the number additional output 
  classes are added or old output classes are removed) will only start if the 
  previous model has a number of output classes different  from the number of 
  output classes entered below. If this is not the case the script will 
  proceed with retraining instead.
  - `Path model for retraining or transfer learning`: path to the model file 
  from which to start retraining or transfer learning.
  - `Enable debug`: if checked the script is run with more detailed output.
  - `Start from scratch`: if checked training will start from scratch, if not
  the script will first look for checkpoints to restart training.
  - `Use augmentations`: if checked training examples are augmented (time and
  pitch shift). This is separate from noise augmentation.
  - `Filter broken audio`: if checked files that are very quiet are removed.
  - `Window size in ms`: the duration (in ms) of the audio example to show. 
  Clips that are longer will automatically be shortened by randomly selecting 
  a section of the correct duration. Clips that are shorter will automatically 
  be zero-buffered.
  - `Maximum number of epochs`: the maximum number of training rounds. 
  Decreasing this number will lower the chance of overfitting, but increase the
  chance of low overall performance. For training, 100-200 rounds is normally
  a good start. 
  - `Early stopping patience in epochs`: how many epochs to accept without 
  improvement. Higher values lead to more learning, but also higher risk of
  overfitting. For training 20 is usually a good start. 
  - `FFT window size`: number of samples to include per window when generating
  the spectrogram.
  - `FFT hop size`: overlap in samples between windows when generating the
  spectrogram.
  - `Frequency minimum`: the minimum frequency (in Hz) of interest, all 
  frequencies lower are not shown to the model, simply be excluding these from
  the image (not using a filter).
  - `Frequency maximum`: the maximum frequency (in Hz) of interest, all 
  frequencies higher are not shown to the model, simply be excluding these from
  the image (not using a filter).
  - `Number of classes`: the number of prediction classes. For detection this
  is 2 (target vs noise). For classification this is the number of classes for
  which you labelled files (often including a noise class).
  - `How many epochs per evaluation`: only touch if you know what you are 
  doing.
  - `Use min max normalization`: only touch if you know what you are doing.
  - `Use jit save`: only touch if you know what you are doing.
  - `Use cuda`: only touch if you know what you are doing.
  - `Batch size`: only touch if you know what you are doing.
  - `Number of worker`: only touch if you know what you are doing.
  - `Learning rate`: only touch if you know what you are doing.
  - `Adam optimizer beta1`: only touch if you know what you are doing.
  - `Learning rate patience in epochs`: only touch if you know what you are 
  doing.
  - `Learning rate decay`: only touch if you know what you are doing.
  - `Frequency compression method`: only touch if you know what you are doing.
  - `Number of frequency bins`: only touch if you know what you are doing.
  - `Resnet`: only touch if you know what you are doing.
  - `Convolutional kernel size`: only touch if you know what you are doing.
  - `Max pooling`: only touch if you know what you are doing.
  
4. Once ready, press `Start training`. The training will start in the 
PowerShell/Terminal console in the background and can be monitored there. 
Training can be stopped by pressing `ctrl + z` or `cmd + z` in the console. 
After closing and restarting the GUI, you can restart training using the 
checkpoints. 

## Retrain a model

If you have the collected new data and want to use a previously trained model 
(e.g. one of the models shared with the publication) as a starting point, you
can use retraining. Retraining assumes the same output classes. If you also 
want to change the output classes continue to the next point instead.

For retraining, you can follow the same steps as for training. You still need
training examples for all classes and you still need to follow the same folder
structure. When selecting the settings, make sure to check 
`Use retraining or transfer learning` and make sure to supply the path to the 
old model. Also make sure that the settings are the same as during the training
of the old model. You might want to reduce the learning rate and number of 
epochs, to make sure the model still remembers some of information.

## Use transfer learning

If you have the collected new data, which contains fewer classes or new 
classes, but still want to use a previously trained model (e.g. one of the 
models shared with the publication) as a starting point, you can use transfer
learning. In this case the entire output layer is removed and replaced with a
new layer with the correct number of output classes, while the encoding part of
the model is kept. 

For transfer learning, you can follow the same steps as for training. You 
still need training examples for all classes and you still need to follow the 
same folder structure. When selecting the settings, make sure to check 
`Use retraining or transfer learning` and make sure to supply the path to the 
old model. Also make sure that the settings are the same as during the training
of the old model. You need to update the number of classes and you might want 
to reduce the learning rate and number of epochs, to make sure the model still 
remembers some of information.

## Prediction

To use a trained model to predict on new audio recordings, you can use the 
prediction tab.

1. Store all the audio files in one folder, without sub folders. Files should
have extension `.wav`. Here is an R script to easily rename files with 
extension `.WAV`: 
<https://github.com/simeonqs/BatSpot_article/blob/main/analysis/code/aspot_wavs_detections.R>

2. Start the GUI. (See installation guide above.)

3. Fill out the required fields, or load a previous config file using the 
`Load settings` button. Predictions are made using a sliding window. Adjacent
windows with the same prediction are merged into a single annotation during 
the translation step.

  - `Path model file`: path to the model file to be used (usually named 
  `ANIMAL-SPOT.pk`)
  - `Path folder with input files`: path to the folder with audio files for
  prediction.
  - `Path folder to store log`: path to the folder where the log file should
  be store. This file contains the console output.
  - `Path folder to store output`: path to the folder where output will be 
  stored. Each file gets a prediction file with extension `_predict_output.log`
  - `Enable debug`: if checked the script is run with more detailed output.
  - `Prediction window size in ms`: the window size use for prediction (in ms).
  - `Prediction hop size in ms`: the overlap between windows (in ms).
  - `Use cuda`: only touch if you know what you are doing.
  - `Visualize output`: only touch if you know what you are doing.
  - `Use jit load`: only touch if you know what you are doing.
  - `Use min max normalization`: only touch if you know what you are doing.
  - `Use latent extraction`: only touch if you know what you are doing.
  - `Batch size`: only touch if you know what you are doing.
  - `Number of worker`: only touch if you know what you are doing.
  
4. Once ready, press `Start prediction`. The training will start in the 
PowerShell/Terminal console in the background and can be monitored there. The 
prediction output are log files, with predictions for every single window. To
translate these into Raven Selection tables, see the next step.

## Translation

To translate the output of the prediction step to Raven Selection tables, you
can use the translation tab. Make sure all prediction file are stored in one
folder without sub folders. 

1. Start the GUI. (See installation guide above.)

2. Fill out the required fields, or load a previous config file using the 
`Load settings` button.

  - `Path folder prediction output`: path to the folder containing the 
  prediction files.
  - `Path folder to store output`: path to the folder where the selection 
  tables will be stored. One file per audio file will be created. If no 
  detections are made, the table will be empty, unless you select 
  `Noise in annotation`.
  - `Noise in annotation`: if checked noise is annotated in the selection table
  as well. 
  - `Prediction threshold`: a number between 0 and 1. The higher the threshold,
  the more 'certain' the model needs to be, before it predicts a detection. A 
  good starting point is 0.5, after which a validation set can be used to 
  balance false positives and false negatives.
  
# FAQ

- After n number of files are processed I get a system error saying that there
was a problem opening a file, but this file is not corrupt, how do I fix this?

  This issue is only known to occur on Linux, by running `ulimit -n 65535` just
  before opening the GUI, you should be able to avoid the system running into 
  the file descriptor limit.

# TODO

- add images
- test all steps
- fix Windows install
- add Linux and Mac install
