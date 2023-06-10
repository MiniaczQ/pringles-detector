from process import process_all
from pathlib import Path
from cv2 import imread, imshow, waitKey, destroyAllWindows
from itertools import islice
from argparse import ArgumentParser
from threading import Thread
from show import imsshow


def print_progress(current: int, total: int, name: str):
    print(f"Processed {current + 1}/{total}: {name}")


def main(source_folder: str, destination_folder: str, preview: int):
    source_folder = Path(source_folder)
    destination_folder = Path(destination_folder)

    total, progress = process_all(source_folder, destination_folder)

    progress = enumerate(progress)

    if preview > 0:
        for_display = []
        for current, destination in islice(progress, preview):
            for_display.append(destination)
            print_progress(current, total, destination)

        show_handle = Thread(target=imsshow, args=[for_display])
        show_handle.start()

    for current, destination in progress:
        print_progress(current, total, destination)

    if preview > 0:
        show_handle.join()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("source_folder")
    parser.add_argument("destination_folder")
    parser.add_argument("-p", "--preview", default=0, type=int)
    args = parser.parse_args()
    main(args.source_folder, args.destination_folder, args.preview)
