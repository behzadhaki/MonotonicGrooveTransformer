Updates 
----

#### Version 0.3

- Added the following keyboard shortcuts (These are enabled only when **CAPS LOCK** is on):
    - *SHORT-CUTS ARE SHOWN ON GUI USING **ORANGE** MARKERS BELOW CORRESPONDING PARAMETERS*
    - **CAPS LOCK** turns keyboard shortcuts on/off
    - **Z, X, C, V, B, N, M** can be used to play a groove in realtime while recording is enabled. 
    The Z has the lowest velocity and M has the highest one
    - **Left** or **Right** arrows  turns metronome volume up or down
    - **Up** or **Down** arrows can be used to increase/decrease volume of the internal clap sound player
    - **P** for Play/Pause 
    - **R** for Record On/Off
    - **Shift+ENTER** Takes snapshot
    - **<** or **>** (shift , or shift .) navigates snapshots
    - **Shift D** Deletes (resets) the input groove 
   
   ######Note
    
   Tested on a Macbook laptop with US International Keyboard Layout. To modify the shortcuts, modify ***midi_map_block.pd*** file 
        
#### Version 0.2
- Added direct controls for changing the model and generation delay from within the pure-data GUI. There are two versions of the trained transformer model: (1) a light one that is smaller, and (2) a larger one that is more computationally intensive. You can go back and forth between these two directly from within the PD patch without re-running the python engine with new parameters. Moreover, if you want to adjust the minimum computation time between the generations, you can now do this directly from PD as well

- Rearranged the GUI to improve the usability
- Added toggle to enable/disable the Numpad for playing grooves via keyboard
- Resolved some bugs related to saving/loading sessions/presets
    
  ###### NOTE
  
  Tested on Pd-0.51-3 on Mac OSX 10.15.7
  
        Requires Cyclone/coll external 
        (can be installed via purr-data: https://agraef.github.io/purr-data/)
        

Installation
----
### Source code
Clone the repository wherever you prefer
    
        git clone https://github.com/behzadhaki/MonotonicGrooveTransformer
        
### Python Environment Setup (using venv) 

Then, open the terminal and navigate to the project folder
    
    cd MonotonicGrooveTransformer
    
Now create a virtual environment for installing the dependencies

    python3 -m venv TorchOSC_venv 

Activate the environment

    source TorchOSC_venv/bin/activate

upgrade pip

    pip3 install --upgrade pip
        
goto https://pytorch.org/get-started/locally/ and get the write pip command for installing torch on your computer. 
(double check the installers using the link)
 
    MAC and Win:
    pip3 install torch torchvision torchaudio

    Linux:
    pip3 install torch==1.10.2+cpu torchvision==0.11.3+cpu torchaudio==0.10.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

Now, install python-osc package

    pip3 install python-osc

### Running Python script

Go to CMC folder, and activate environment
    
    source TorchOSC_venv/bin/activate
    
Finally, run the python script

    python run_generative_engine.py  --py2pd_port 1123 --pd2py_port 1415 --wait 2 --model light_version
    
