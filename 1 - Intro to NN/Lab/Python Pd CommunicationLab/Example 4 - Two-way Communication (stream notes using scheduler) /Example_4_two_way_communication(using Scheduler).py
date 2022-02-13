import queue
import time
import threading


from base_models import OscReceiver, NoteGenerator, NoteSequenceQueueToPlaybackScheduler


class GenerativeServer:
    def __init__(configs):
        pass


if __name__ == '__main__':

    ticks_queue = queue.Queue(maxsize=0)
    playback_sequence_queue = queue.Queue(maxsize=0)

    print("tasks to do", playback_sequence_queue.unfinished_tasks)

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
                "playback_sequence_queue": playback_sequence_queue
            },

        # configs for the thread generating new sequence (i.e. Mock AI Generator)
        "generation_configs":
            {
                "generate_thread_Event": generate_thread_Event,
                "playback_sequence_queue": playback_sequence_queue,
                "sequence_length": 16,                          # 16
                "generation_time": .1,                          # The time between generating
                "pitch_range": (40, 52),                        # pitch range
                "velocity_range": (78, 80),                     # velocity range (0, 127)
                "duration_range": (30, 200),                    # note length in ms
                "onset_difference_range": (100, 100),          # note length in ms
                "grace_time_before_playback": 2,                # (sec) time to wait before playing sequence
                "quit_event": quit_event
            }


    }

    print("here")

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

    note_generator = NoteGenerator(configs["generation_configs"], )
    note_generator.start()

    playback_scheduler = NoteSequenceQueueToPlaybackScheduler(
        note_sequence_queue=playback_sequence_queue,
        note_to_pd_configs=configs["note_to_pd"],
        quit_event=quit_event)
    playback_scheduler.start()


    time.sleep(1000)

    # while not ticks_queue.empty():
    #   print("ticks received so far: tick {}".format(ticks_queue.get()))

    while not playback_sequence_queue.empty():
        print("Note received: {}".format(playback_sequence_queue.get()))

    configs["ticks_from_pd"]["quit_event"].set()


