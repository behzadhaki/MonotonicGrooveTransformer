import queue
import time
import threading

from model.base_osc import OscReceiver, NoteGenerator, NoteSequenceQueueToPdSender


if __name__ == '__main__':

    playback_sequence_queue = queue.Queue(maxsize=0)
    sequence_generation_destination_queue = queue.Queue(maxsize=0)

    generate_melody_thread_Event = threading.Event()

    def actions_handler(address, *args):
        # handler for messages starting with /clock
        address = address.split("/actions")[-1]
        print("address", address)
        if "/generate_left_melody" in address:
            sequence_generation_destination_queue.put(("Left_Melody", args[0]))
            generate_melody_thread_Event.set()
            print("left melody note_generator notified!")
        if "/generate_right_melody" in address:
            sequence_generation_destination_queue.put(("Right_Melody", args[0]))
            generate_melody_thread_Event.set()
            print("right melody note_generator notified!")
        if "/generate_interpolation" in address:
            sequence_generation_destination_queue.put(("Interpolated_Melody", args[0]))
            generate_melody_thread_Event.set()
            print("interpolation note_generator notified! --> distance between Left/right: {}".format(args[0]))

    quit_event = threading.Event()

    configs = {

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
                "generate_thread_Event": generate_melody_thread_Event,
                "sequence_generation_destination_queue": sequence_generation_destination_queue,
                "playback_sequence_queue": playback_sequence_queue,
                "measure_duration": 16,                          # 16 16th notes per measure
                "n_classes_per_step": 14,                       # number of classes for each time step 0:
                                                                # hold 1: rest 2-13: A to G#
                "generation_time": .1,                          # The time between generating
                "quit_event": quit_event,
                "min_num_notes": 1,                             # min number of notes required in the
                                                                # generated melodies (should be >=1)
                "bpm": 120,                                     # BPM
                "vae_checkpoint": "checkpoints/epoch_800.pt"
            }


    }

    actions_message_receiver = OscReceiver(
        ip=configs["actions_from_pd"]["ip"],
        receive_from_port=configs["actions_from_pd"]["port"],
        quit_event=configs["actions_from_pd"]["quit_event"],
        address_list=configs["actions_from_pd"]["address_list"],
        address_handler_list=configs["actions_from_pd"]["address_handler_list"]
    )
    actions_message_receiver.start()

    note_generator = NoteGenerator(configs["generation_configs"])
    note_generator.start()

    note_to_pd_sender = NoteSequenceQueueToPdSender(
        note_sequence_queue=playback_sequence_queue,
        note_to_pd_configs=configs["note_to_pd"],
        quit_event=quit_event)
    note_to_pd_sender.start()


    time.sleep(1000)

    # while not ticks_queue.empty():
    #   print("ticks received so far: tick {}".format(ticks_queue.get()))

    while not playback_sequence_queue.empty():
        print("Note received: {}".format(playback_sequence_queue.get()))

    configs["ticks_from_pd"]["quit_event"].set()


