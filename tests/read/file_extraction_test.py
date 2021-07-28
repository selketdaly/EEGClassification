import unittest
from unittest.mock import MagicMock, call
from EEGClassification.read import file_extraction as fe
import os
import tarfile


class ExtractTarsTest(unittest.TestCase):
    directory = os.path.dirname(os.path.dirname(__file__))
    target_directory = directory + "/data"

    def test_open_tar_file(self):
        fe.extract_archive = MagicMock(name='method')
        tarfile.open = MagicMock(name='method')
        fe.extract_tars(directory=self.directory, target_directory=self.target_directory)
        tarfile.open.assert_called_once_with(name=self.directory + "/TestData.tar")

    def test_extract_from_tar_file(self):
        fe.extract_archive = MagicMock(name='method')
        mock_tar = MagicMock(return_value='MockedTar')
        tarfile.open = MagicMock(name='method', return_value=mock_tar)
        fe.extract_tars(directory=self.directory, target_directory=self.target_directory)
        fe.extract_archive.assert_called_once_with(archive=mock_tar.__enter__(), target_directory=self.target_directory)


class ExtractArchiveTest(unittest.TestCase):
    directory = os.path.dirname(os.path.dirname(__file__))
    target_directory = directory + "/data"
    mock_tar = MagicMock(return_value='MockedTar')

    def setUp(self) -> None:
        self.mock_tar.getnames = MagicMock(name='method', return_value=["file1", "file2"])

    def test_extract_each_file_from_tar(self):
        fe.extract_archive(archive=self.mock_tar, target_directory=self.target_directory)
        self.mock_tar.extract = MagicMock(name='method')
        calls = [call.extract(member="file1", path=self.target_directory), call.extract(member="file2", path=self.target_directory)]
        self.mock_tar.assert_has_calls(calls, any_order=True)

    def test_do_not_extract_if_already_extracted(self):
        os.path.exists = MagicMock(name='method', return_value=True)
        fe.extract_archive(archive=self.mock_tar, target_directory=self.target_directory)
        self.mock_tar.extract = MagicMock(name='method')
        calls = [call.extract("file1", path=self.target_directory), call.extract("file2", path=self.target_directory)]
        self.mock_tar.assert_not_called()


if __name__ == '__main__':
    unittest.main()
