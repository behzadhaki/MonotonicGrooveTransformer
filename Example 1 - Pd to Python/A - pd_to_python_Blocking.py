import time

#!pip install python-osc
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher

if __name__ == '__main__':
    # used to quit osc_receiver
    ##################################################################
    ##################################################################
    ########### OSC MESSAGE HANDLERS #################################
    ##################################################################
    ##################################################################
    #  Values received from sliders or numboxes will be stored/updated
    #       in the dedicated lists: slider_values and num_box_values
    #       if you need more than 10 sliders, increase the length of
    #       the default lists in lines 101-102
    #
    #  The methods slider_message_handler and num_box_message_handler
    #       are in charge of updating the slider_values and num_box_values
    #       lists using the corresponding received osc messages

    n_sliders = 10
    n_num_boxes = 10
    quitFlag = False

    # Lists for storing slider and nbox values
    slider_values = [0 for _ in range(n_sliders)]
    num_box_values = [0 for _ in range(n_num_boxes)]
    quitFlag = [quitFlag]

    # connection parameters
    ip = "127.0.0.1"
    receiving_from_port = 1415

    # dispatcher is used to assign a callback to a received osc message
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

    def quit_message_handler(address, *args):
        quitFlag[0] = True
        print("QUITTING!")

    # pass the handlers to the dispatcher
    dispatcher.map("/slider*", slider_message_handler)
    dispatcher.map("/nbox*", num_box_message_handler)
    dispatcher.map("/quit*", quit_message_handler)


    # you can have a default_handler for messages that don't have dedicated handlers
    def default_handler(address, *args):
        print(f"No action taken for message {address}: {args}")
    dispatcher.set_default_handler(default_handler)

    # python-osc method for establishing the UDP communication with pd
    server = BlockingOSCUDPServer((ip, receiving_from_port), dispatcher)

    ##################################################################
    ##################################################################
    ########### MAIN CODE HERE #######################################
    ##################################################################
    ##################################################################

    while (quitFlag[0] is False):
        server.handle_request()
        print("sliders: ", slider_values, "nbox: ", num_box_values)