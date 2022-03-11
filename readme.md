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
    
