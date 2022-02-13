import threading
import random
import time

from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient


class OscReceiver(threading.Thread):

    def __init__(self, ip, receive_from_port, quit_event, address_list=["/clock*"], address_handler_list=[None]):
        """
        Constructor for OSC_SENDER CLASS

        :param ip: ip address of client ==> 127.0.0.1 (for local host/ inter app communication on same machine)
        :param receive_from_port: the port on which python listens for incoming data
        """
        super(OscReceiver, self).__init__()
        self.setDaemon(True) # don't forget this line, otherwise, the thread never ends

        self.ip = ip
        self.receiving_from_port = receive_from_port

        self.listening_thread = None

        self.dispatcher = Dispatcher()

        for ix, address in enumerate(address_list):
            self.dispatcher.map(address, address_handler_list[ix])

        self.dispatcher.set_default_handler(self.default_handler)

        self.server = BlockingOSCUDPServer((self.ip, self.receiving_from_port), self.dispatcher)
        #self.server.request_queue_size = 0
        self.quit_event = quit_event

    def run(self):
        print("running --- waiting for data")
        count = 0
        while not self.quit_event.is_set():

            self.server.handle_request()
            #count = (count+1) #% 8
            #print("count {}".format(count))

    def default_handler(self, address, *args):
        # handler for osc messages with no specific defined decoder/handler
        print(f"DEFAULT {address}: {args}")

    def get_ip(self):
        return self.ip

    def get_receiving_from_port(self):
        return self.receiving_from_port

    def get_server(self):
        return self.server

    def change_ip_port(self, ip, port):
        self.ip = ip
        self.receiving_from_port = port
        self.server = BlockingOSCUDPServer(self.ip, self.receiving_from_port)


class OscSender(threading.Thread):
    def __init__(self, ip, sending_to_port, ticks_queue, playback_sequence_queue):
        """
        Constructor for OSC_SENDER CLASS

        :param ip: ip address of client ==> 127.0.0.1 (for local host/ inter app communication on same machine)
        :param sending_to_port: the port on which pure data listens for incoming data
        """
        super(OscSender, self).__init__()
        self.setDaemon(True)

        self.ip = ip
        self.sending_to_port = sending_to_port

        self.ticks_queue = ticks_queue
        self.playback_sequence_queue = playback_sequence_queue

        self.client = SimpleUDPClient(self.ip, self.sending_to_port)

    def run(self):
        (start_tick, pitch, velocity, duration) = (None, None, None, None)
        while True:
            if start_tick is None:  # if no note to play, wait for a new note
                (start_tick, pitch, velocity, duration) = self.playback_sequence_queue.get()
            else:                   # if note available, wait for the correct tick time and send the note to pd
                current_tick = self.ticks_queue.get()
                if current_tick == start_tick:
                    self.send_to_pd(["/note/duration", "/note/velocity", "/note/pitch"],
                                    [duration, velocity, pitch])
                    #print(f"note played at tick position {current_tick}  ")
                    (start_tick, pitch, velocity, duration) = (None, None, None, None)
            #time.sleep(1)

    def send_to_pd(self, message_parameters, message_values):
        """
        sends a list of messages to pd
        Note 1: Messages are sent in the same order as presented in the lists
        Note 2: ALWAYS USE LISTS EVEN IF SENDING A SINGLE PARAM/VALUE

        :param message_parameters: list of str: example ["/note/pitch", /note/duration"]
        :param message_values: list of ints, floats: example [53, 1000]
        """

        if len(message_parameters) != len(message_values):
            raise ValueError("The number of message_types do not match the values")

        else:
            for ix, param in enumerate(message_parameters):
                self.client.send_message(param, message_values[ix])

    def get_ip(self):
        return self.ip

    def get_sending_to_port(self):
        return self.sending_to_port

    def get_client(self):
        return self.client

    def change_ip_port(self, ip, port):
        self.ip = ip
        self.sending_to_port = port
        self.client = SimpleUDPClient(self.ip, self.sending_to_port)


class NoteGenerator(threading.Thread):
    def __init__(self, generation_configs, bpm, ppq):

        super(NoteGenerator, self).__init__()
        self.setDaemon(True)  # don't forget this line, otherwise, the thread never ends

        self.generation_configs = generation_configs

        self.bpm = bpm
        self.ppq = ppq

        self.tick_duration = 60000/(self.bpm*self.ppq)

    def run(self):
        # note format (start_tick, pitch, velocity, duration)
        # self.generation_configs["generate_thread_condition"].acquire()
        while not self.generation_configs["quit_event"].is_set():
            self.generation_configs["generate_thread_Event"].wait()
            self.generation_configs["generate_thread_Event"].clear()
            sequence = []
            start_time = 0
            for i in range(self.generation_configs["sequence_length"]):
                print(f"Generating {i}th note!!!!")
                velocity = int(random.randrange(*self.generation_configs["velocity_range"]))
                duration = int(random.randrange(*self.generation_configs["duration_range"]))
                pitch = int(random.randrange(*self.generation_configs["pitch_range"]))

                start_time = start_time + int(random.randrange(
                    *self.generation_configs["onset_difference_range"]))

                start_tick = int(start_time/self.tick_duration)

                note = (start_tick, pitch, velocity, duration)

                self.generation_configs["playback_sequence_queue"].put(note)

                time.sleep(self.generation_configs["generation_time"])

                print(note)

    def update_bpm(self, bpm):
        self.bpm = bpm
        self.tick_duration = 60000/(self.bpm*self.ppq)

    def update_ppq(self, ppq):
        self.ppq = ppq
        self.tick_duration = 60000/(self.bpm*self.ppq)