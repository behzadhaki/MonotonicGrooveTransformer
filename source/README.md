Internal Synths
------


1. If you want to use a pd based synth for synthesizing input to system, I have placed a copy of the amazingly beautiful sounding EP-MK1 by Mike Moreno:
    
    
    https://github.com/MikeMorenoDSP/EP-MK1
    

Please read the readme in the EP-MK1_Pd_Standalone Patch provided by the author of this patch. 
This patch has been only modified such that instead of receiving midi in, the notes are received via a receive object,
named [receive notes_to_EP_MK1] 


![](.README_images/before_after.png)


2. Samples for the internal drum module (if needed) have been obtained from:

   
    https://github.com/crabacus/the-open-source-drumkit