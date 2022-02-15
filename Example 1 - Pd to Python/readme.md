# Python to Pd Communication
Here, we will discuss how to send data from pd to python using OSC protocol. 

There are two examples here:

- (A): Receiving and processing data in a single loop
- (B): Receiving data in the background (in a separate thread) and processing data continuosly

Sending data out of Pd
----- 

To discuss Py to Pd communication, we will use the following pd patch:

![plot](./images/pdPatch.png) 

The objective here is to use sliders and number boxes to modify some variables in our python script. 

To do so, we start with packing the value of the sliders and boxes into OSC messages which will be sent over to python.

    
       # format of osc messages for the N-th slider
       /slider/N Slider_Value
       
       # format of osc messages for the N-th number box
       /nbox/N NumberBox_Value

For packing the message (similar to above example), we can use Pd's builtin **oscformat** method.
This method basically adds a given V to a required address. 
This message should then be prepended with a **send** instruction.
Finally, to send the prepared messages to python, we use the **[netsend  ]** method in pd. 

Receiving data in the Python script
------
## Example A
To receive OSC messages in python, we use the **pythonosc** library.

    pip install python-osc
    
From this library, we will use two methods:

    from pythonosc.osc_server import BlockingOSCUDPServer
    from pythonosc.dispatcher import Dispatcher

**BlockingOSCUDPServer** is used to establish a UDP connection with pd. To use this method, an **IP** address and **Port** number should be specified. 

    server = BlockingOSCUDPServer((ip, receiving_from_port), dispatcher)

Moreover, we can use the **Dispatcher** method to define what sort of action to be taken for a given received message. 
In these examples, the only action required is to update a number of local lists with the received slider/number box values received from pd.


# dispatcher is used to assign a callback to a received osc message
    
    n_sliders = 10
    n_num_boxes = 10

    # Lists for storing slider and nbox values
    slider_values = [0 for _ in range(n_sliders)]
    num_box_values = [0 for _ in range(n_num_boxes)]
    
    # in other words the dispatcher routes the osc message to the right action using the address provided
    dispatcher = Dispatcher()

    # define the handler for messages starting with /slider/[slider_id]
    def slider_message_handler(address, *args):
        slider_id = address.split("/")[-1]
        slider_values[int(float(slider_id))] = args[0]

    # define handler for messages starting with /nbox/[nbox_id]
    def num_box_message_handler(address, *args):
        nbox_id = address.split("/")[-1]
        num_box_values[int(float(nbox_id))] = args[0]

    # pass the handlers to the dispatcher
    dispatcher.map("/slider*", slider_message_handler)
    dispatcher.map("/nbox*", num_box_message_handler)

 Once the communication is set up and the actions for handling the messages are defined, we can continuously wait for messages to arrive from pd
 
    while (1):
        server.handle_request()
        
        // take_some_action_using_data here
         print("sliders: ", slider_values, "nbox: ", num_box_values)    

 The **handle_request** will basically call the correct handler for a received message as soon as available.
 
 Discussion
 ---
 If we wait for receiving messages in a loop (**Example A**), the program will only be able to carry out some action using the values, whenever there is a message available. 
 In other words, if no messages is sent over to python, the program will be indefinitely paused on **server.handle_request()**
 
 In case we want to continuously generate content in the main loop, we will need to make sure that the reception/handling of the osc message is done in the background. 
 This can be easily done using a separate thread on which the data is received and their corresponding variables are updated. (**See Example 2**)