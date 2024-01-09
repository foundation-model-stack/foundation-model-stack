import sys
from collections import UserDict

import pyarrow as pa
import urllib3
from pyarrow.fs import FileSystem, FileType, LocalFileSystem, S3FileSystem
from torch.utils.data import Dataset, IterableDataset

from fms.datasets import DatasetStateDictMixin


class _ArrowFileData(UserDict):
    def __init__(self, fs: FileSystem, path: str):
        self.fs = fs
        self.path = path

    def __getitem__(self, idx: int):
        with self.fs.open_input_file(self.path) as file:
            reader = pa.ipc.open_file(file)
            print(self.path, file, idx)
            return reader.get_batch(idx)

    def __iter__(self):
        with self.fs.open_input_file(self.path) as file:
            reader = pa.ipc.open_file(file)
            for i in range(reader.num_record_batches):
                yield reader.get_batch(i)

    def __len__(self):
        with self.fs.open_input_file(self.path) as file:
            reader = pa.ipc.open_file(file)
            return reader.num_record_batches


class ArrowFilesDataset(IterableDataset, DatasetStateDictMixin):
    """
    Creates a dataset from a path to a directory of arrow files, either in
    S3/COS or a local file system.

    uri: s3://endpoint_host/path/to/files or file:///path/to/files
    step: for distributed training, use rank+1
    start_idx: for re-starting training, pick up where left off
    """

    def __init__(self, uri: str, start_idx: int = 0, step: int = 1):
        self.uri = uri
        self.start_idx = start_idx
        self.step = step
        self._initialize()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._initialize()

    def _initialize(self):
        url = urllib3.util.parse_url(self.uri)
        path = url.path
        file_system = None
        if url.scheme == "file":
            file_system = LocalFileSystem()
        elif url.scheme == "s3":
            if url.host is not None:
                endpoint = "https://" + url.host
                file_system = S3FileSystem(endpoint_override=endpoint)
            else:
                file_system = S3FileSystem()
            if path[0] == "/":
                path = path[1:]
        else:
            raise ValueError(f"Unsupported scheme {url.scheme}")

        files = file_system.get_file_info(pa.fs.FileSelector(path))
        files = [
            _ArrowFileData(file_system, f.path)
            for f in files
            if f.type != FileType.Directory
        ]

        self._files = files
        self.length = sum([len(f) for f in files])

        file_offset = self.start_idx
        while file_offset > len(self._files[0]):
            file_offset -= len(self._files[0])
            self._files = self._files[1:]
        self.file_offset = file_offset
        self.file_offset += self.step

    def __iter__(self):
        for file in self._files:
            remainder = (len(file) - self.file_offset) % self.step
            remainder = self.step - remainder
            for i in range(self.file_offset, len(file), self.step):
                self.start_idx += self.step
                yield file[i]
            self.file_offset = remainder


if __name__ == "__main__":
    if not len(sys.argv) == 2:
        print("Usage: python -m fms.datasets.arrow <path>")
        sys.exit(1)
    ds = ArrowFilesDataset(sys.argv[1])
    for i, batch in enumerate(ds):
        print(i, batch)
        if i > 20:
            break
