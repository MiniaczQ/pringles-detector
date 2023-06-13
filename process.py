from cv2 import imread, imwrite
from pathlib import Path
from os import makedirs
from typing import Tuple, List, Generator
from detect import detect
from shutil import rmtree


def process_one(source: Path, destination: Path) -> Path:
    """
    Load, performs detection and saves a single image.
    """
    assert source.is_file()
    assert not destination.exists()

    image = imread(str(source))
    image = detect(image)
    imwrite(str(destination), image)

    return destination


def process_many(
    sources_and_destinations: List[Tuple[Path, Path]],
) -> Generator[Path, None, None]:
    """
    Load, performs detection and saves many images.
    """
    for source, destination in sources_and_destinations:
        yield process_one(source, destination)


def process_all(
    source_folder: Path, destination_folder: Path
) -> Tuple[int, Generator[Path, None, None]]:
    """
    Load, performs detection and saves all images from source folder into the destination folder.
    """
    assert source_folder.is_dir()

    if destination_folder.exists():
        rmtree(destination_folder)
    makedirs(destination_folder)

    sources_and_destinations = list(
        [
            (source, destination_folder.joinpath(source.name))
            for source in source_folder.iterdir()
        ]
    )

    return (len(sources_and_destinations), process_many(sources_and_destinations))
