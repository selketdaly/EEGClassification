import os
import tarfile
from tarfile import TarFile


def extract_archive(archive: TarFile, target_directory: str) -> None:
    print("Extracting archives")
    for name in archive.getnames():
        if os.path.exists(os.path.join(target_directory, name)):
            print(name, "already exists")
        else:
            archive.extract(member=name, path=target_directory)


extract_archive.__doc__ = "Extracts the archive, ignoring files which have already been extracted"


def extract_tars(directory: str, target_directory: str) -> None:
    print("Loading tars")
    archives = [name for name in os.listdir(path=directory) if name.endswith("tar")]
    for archive_name in archives:
        with tarfile.open(name=directory + "/" + archive_name) as archive:
            extract_archive(archive=archive, target_directory=target_directory)


extract_tars.__doc__ = "Looks for tar files in the project root to extract"
