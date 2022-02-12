import queue
import time
import threading


from base_models import OscSender, OscReceiver, NoteGenerator


if __name__ == '__main__':

    ticks_queue = queue.Queue(maxsize=0)
    playback_sequence_queue = queue.Queue(maxsize=0)


    def clock_message_handler(address, *args):

        # handler for messages starting with /clock
        # print("Clock Received")
        #print("address: ", address)
        # print("args: ", args)
        ticks_queue.put(args[0])

    generate_thread_Event = threading.Event()

    def actions_handler(address, *args):
        # handler for messages starting with /clock
        address = address.split("/actions")[-1]
        if "/generate" in address:
            generate_thread_Event.set()
            print("note_generator notified!")

    quit_event = threading.Event()

    configs = {

        "bpm": 120,         # BPM
        "PPQ": 192,         # Parts per quarter

        # configs for the thread receiving midi ticks from pd
        "ticks_from_pd":
            {
                "ip": "127.0.0.1",
                "port": 1234,
                "quit_event": quit_event,
                "address_list": ["/clock*"],
                "address_handler_list": [clock_message_handler]
            },

        # configs for the thread receiving action messages  from pd
        "actions_from_pd":
            {
                "ip": "127.0.0.1",
                "port": 1235,
                "quit_event": quit_event,
                "address_list": ["/actions*"],  # some other actions to be implemented: bpm and PPQ updater
                "address_handler_list": [actions_handler]
            },

        # configs for the thread sending generated notes to pd
        "note_to_pd":
            {
                "ip": "127.0.0.1",
                "port": 1240,
                "quit_event": quit_event,
            },

        # configs for the thread generating new sequence (i.e. Mock AI Generator)
        "generation_configs":
            {
                "generate_thread_Event": generate_thread_Event,
                "playback_sequence_queue": playback_sequence_queue,
                "sequence_length": 16,
                "generation_time": .4,
                "pitch_range": (42, 42+24),
                "velocity_range": (78, 80),                     # max range (0, 127)
                "duration_range": (800, 2000),                   # note length in ms
                "onset_difference_range": (1000, 1001),           # note length in ms
                "quit_event": quit_event
            }


    }


    clock_tick_receiver = OscReceiver(
        ip=configs["ticks_from_pd"]["ip"],
        receive_from_port=configs["ticks_from_pd"]["port"],
        quit_event=configs["ticks_from_pd"]["quit_event"],
        address_list=configs["ticks_from_pd"]["address_list"],
        address_handler_list=configs["ticks_from_pd"]["address_handler_list"]
    )
    clock_tick_receiver.start()

    actions_message_receiver = OscReceiver(
        ip=configs["actions_from_pd"]["ip"],
        receive_from_port=configs["actions_from_pd"]["port"],
        quit_event=configs["actions_from_pd"]["quit_event"],
        address_list=configs["actions_from_pd"]["address_list"],
        address_handler_list=configs["actions_from_pd"]["address_handler_list"]
    )
    actions_message_receiver.start()

    note_generator = NoteGenerator(configs["generation_configs"], configs["bpm"], configs["PPQ"])
    note_generator.start()

    NoteOscSender = OscSender(
        ip=configs["note_to_pd"]["ip"],
        sending_to_port=configs["note_to_pd"]["port"],
        ticks_queue=ticks_queue,
        playback_sequence_queue=playback_sequence_queue)

    NoteOscSender.start()

    time.sleep(1000)

    """while not playback_sequence_queue.empty():
        print("Note received: {}".format(playback_sequence_queue.get()))"""

    configs["ticks_from_pd"]["quit_event"].set()


    print("here8")
