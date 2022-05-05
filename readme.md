Updates 
----

#### Version 0.7
- Cleaned up the implementation for setting the slider ut and vel values based on incoming velocities. The send destinations are now dynamically adjusted 
- From now on, by default, the groove is updated only when the velocity is changed (if you need to update on utiming change
as well, deactivate the toggle labeled ***"Update_Only_On_Vel_Change"***)
- Two graphs added
    1. Onset_Tracker: showing all events over time
    2. Onset_Histogram: showing the number of triggers at each time-step  


#### Version 0.6
- Added a hradio object to GUI to specify the starting time-step of the sequence
- The sequence starting point can now be adjusted. 

#### Version 0.5
- Incorporated an internal drum synthesizer for easier testing. The drum engine is a hybrid sample/synthesis patch.
        
        The samples were obtained from:
        https://github.com/crabacus/the-open-source-drumkit
        
        The synths were incroporated from mymembrane~ instruments developed by Mike Moreno DSP
        https://github.com/MikeMorenoDSP/pd-mkmr/tree/master/instruments 
        
- Incorporated two internal synthesizers using the patches developed by ***Mike Moreno DSP***:

        santur obtained from https://github.com/MikeMorenoDSP/pd-mkmr/tree/master/instruments
        EP-MK1 obtained from https://github.com/MikeMorenoDSP/EP-MK1
 
  ###### Note
  These patches have been slightly modified so as to use internal midi routes without using any virtual midi paths within the patch
  
- From this version on, a separate midi out channel can be specified to send out two notes corresponding to beat/bar positions. 
This can be used for debugging/synchronizing external clock receivers
    
    
#### Version 0.4

- Improved the performance by updating the logic involved in repainting of events in plot_trigger.pd
 
    - Instead of resetting the entirety of a timestep before re-plotting, now only the old value is reset. 
    This wont be able to remove manually drawn events in the groove canvas. As a result, a repaint button was added to force repaint the groove with the actual vels and utimings 
 
 - Fixed clock out issue: Removed channel specifier for sending clock out as specifying it wouldn't 
     allow the DAW to receive transport information sent out of pure data's **[midisystemrealtime]** module.

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
- Clock is now sent only through channel 16 of midi out. 

- Midi controllers with infinite encoders or slider/potentiometers (0-127) can be used for controlling generation params.
The mappings can also be saved so as to import in future projects as well 

At each beat/bar a single note is also sent out to check synchronization over time
   ######Note
    
   Tested on a Macbook laptop with US International Keyboard Layout. To modify the shortcuts, modify ***keyboard_interface.pd*** file 
        
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
    
