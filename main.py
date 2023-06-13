from process import process_all
from pathlib import Path
from cv2 import imread, imshow, waitKey, destroyAllWindows
from itertools import islice
from argparse import ArgumentParser
from threading import Thread
from show import imsshow
from datetime import datetime


def print_progress(current: int, total: int, name: str):
    """
    Displays progress of the detection.
    """
    print(f"Processed {current}/{total}: {name}")


def main(source_folder: str, destination_folder: str, preview: int):
    """
    Main application.
    """
    source_folder = Path(source_folder)
    destination_folder = Path(destination_folder).joinpath(
        datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    )

    # Generator which performs detection on each `next` call
    total, progress = process_all(source_folder, destination_folder)

    # Collects first preview images for display
    if preview > 0:
        for_display = []
        for current, destination in enumerate(islice(progress, preview)):
            for_display.append(destination)
            print_progress(1 + current, total, destination)

        # Starts display on another thread to not block further detection.
        show_handle = Thread(target=imsshow, args=[for_display])
        show_handle.start()

    # Completes the detection on other images.
    for current, destination in enumerate(progress):
        print_progress(1 + preview + current, total, destination)

    # Awaits preview display to stop.
    if preview > 0:
        show_handle.join()


if __name__ == "__main__":
    # Argument parsing
    parser = ArgumentParser(
        "Pringles logo detector",
        description="Detects logo of the Pringles brand in images.",
    )
    parser.add_argument("source_folder", help="Folder with input images.")
    parser.add_argument(
        "destination_folder",
        help="Folder where images with marked detections will go to.",
    )
    parser.add_argument(
        "-p",
        "--preview",
        default=0,
        type=int,
        help="How many first images to preview during processing.",
    )
    args = parser.parse_args()
    main(args.source_folder, args.destination_folder, args.preview)
