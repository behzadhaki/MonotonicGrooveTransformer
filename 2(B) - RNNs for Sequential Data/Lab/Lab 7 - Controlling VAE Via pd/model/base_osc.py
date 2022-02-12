import threading
import random
import time

from datetime import datetime, timedelta

from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient

from apscheduler.schedulers.background import BackgroundScheduler

from model.Melody_VAE import VAE
import torch

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


class OscSender():
    def __init__(self, sender_configs):
        """
        Constructor for OSC_SENDER CLASS

        :param ip: ip address of client ==> 127.0.0.1 (for local host/ inter app communication on same machine)
        :param sending_to_port: the port on which pure data listens for incoming data
        """

        self.sender_configs = sender_configs

        self.ip = self.sender_configs["ip"]
        self.sending_to_port = self.sender_configs["port"]

        self.playback_sequence_queue = self.sender_configs["playback_sequence_queue"]

        self.client = SimpleUDPClient(self.ip, self.sending_to_port)

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
    def __init__(self, generation_configs, generation_destination="Left_Melody"):

        super(NoteGenerator, self).__init__()
        self.setDaemon(True)  # don't forget this line, otherwise, the thread never ends

        self.generation_configs = generation_configs

        self.vae = VAE()      # Create an instance of the vae model
        self.vae.load_state_dict(torch.load(generation_configs["vae_checkpoint"]))    # load pretrained parameters
        self.vae.eval()      # Set to eval to signal the model that we are just using the model for inference

        # initiate the embeddings
        self.left_melody_z_embedding = torch.zeros(1, 20)
        self.right_melody_z_embedding = torch.zeros(1, 20)
        self.interpolated_melody_z_embedding = torch.zeros(1, 20)

        # initiate the interpolation factor
        self.interpolation_factor = 0.0

        # initiate the generation_destination
        self.generation_destination = "Left_Melody"    # Left_Melody, Right_Melody or Interpolated_Melody


    def run(self):
        # note format (start_tick, pitch, velocity, duration)
        # self.generation_configs["generate_thread_condition"].acquire()
        while not self.generation_configs["quit_event"].is_set():

            # wait for generation request via generate_thread_Event
            self.generation_configs["generate_thread_Event"].wait()
            self.generation_configs["generate_thread_Event"].clear()

            # get the generation type/destination
            if not self.generation_configs["sequence_generation_destination_queue"].empty():
                # Messages in this queue are a tuple the form of ("Left_Melody", 0),
                # ("Right_Melody", 0), ("Interpolated_Melody", Interpolation factor)
                temp = self.generation_configs["sequence_generation_destination_queue"].get()
                self.generation_destination = temp[0]
                if temp[0] == "Interpolated_Melody":
                    self.interpolation_factor = temp[1]
                    # print("interpolation_factor updated!!!", self.interpolation_factor )

                # print("UPDATED GENERATION DESTINATIONS!!!!", self.generation_destination)

            if self.generation_destination == "Left_Melody" or\
                    self.generation_destination == "Right_Melody":

                # Generate a random sequence
                starts, notes, durations, sequence, sample_z = self.vae.generate_random_sequence(
                    min_num_notes=self.generation_configs["min_num_notes"])

                if self.generation_destination == "Left_Melody":
                    self.left_melody_z_embedding = sample_z
                else:
                    self.right_melody_z_embedding = sample_z

                self.put_notes_to_queue(self.generation_destination, starts, notes, durations, sequence)

            # Generate interpolated melody
            self.interpolated_melody_z_embedding = (self.right_melody_z_embedding - self.left_melody_z_embedding )*\
                                                   float(self.interpolation_factor) + self.left_melody_z_embedding

            # print("distance", self.right_melody_z_embedding - self.left_melody_z_embedding )
            # print("scaled distance", (self.right_melody_z_embedding - self.left_melody_z_embedding )*self.interpolation_factor)

            # print("left embed", self.left_melody_z_embedding)
            # print("right embed", self.right_melody_z_embedding)
            # print("interpolated embed", self.interpolated_melody_z_embedding)

            interpolated_sequence_logits = self.vae.decode(self.interpolated_melody_z_embedding)
            # convert logits to most probable sequence and individual lists of starts, notes and durations
            starts, notes, durations, sequence = self.vae.logits_to_starts_notes_durations(
                interpolated_sequence_logits)

            self.put_notes_to_queue("Interpolated_Melody", starts, notes, durations, sequence)

    def put_notes_to_queue(self, destination, starts, notes, durations, sequence):
        print("Sequence generated for {}: {}".format(destination, sequence))
        for i in range(len(starts)):
            note = (destination, starts[i], notes[i], durations[i])
            self.generation_configs["playback_sequence_queue"].put(note)

    def update_bpm(self, bpm):
        self.bpm = bpm
        self.tick_duration = 60000/(self.bpm*self.ppq)

    def update_ppq(self, ppq):
        self.ppq = ppq
        self.tick_duration = 60000/(self.bpm*self.ppq)


class NoteSequenceQueueToPdSender(threading.Thread):
    def __init__(self, note_sequence_queue, note_to_pd_configs, quit_event):
        super(NoteSequenceQueueToPdSender, self).__init__()
        self.setDaemon(True)  # don't forget this line, otherwise, the thread never ends

        self.note_sequence_queue = note_sequence_queue

        self.note_to_pd_configs = note_to_pd_configs

        self.quit_event = quit_event

        self.OscSender = OscSender(self.note_to_pd_configs)

        self.playback_scheduler = BackgroundScheduler(daemon=True)
        self.playback_scheduler.start()

        # print("scheduler running? :", self.playback_scheduler.running)

    def run(self):

        while True:

            # Note format: (start_time, pitch, velocity, duration)
            note = self.note_sequence_queue.get()

            # print("Note gotten: ", note)

            # initial note format (start_tick, pitch, velocity, duration)
            note = [*note]  # convert tuple to list and extract starting_index, pitch_class, duration in steps

            self.OscSender.send_to_pd([f"/note/{note[0]}/pitch", f"/note/{note[0]}/duration", f"/note/{note[0]}/startTime"],
                                      [note[2], note[3], note[1]])

