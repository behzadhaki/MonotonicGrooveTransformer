Installation
----
### Source code
Clone the repository wherever you prefer

### Python Environment Setup (using venv) 

Then, open the terminal and navigate to the lab folder
    
    cd "CMC_SMC/Lab"
    
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

