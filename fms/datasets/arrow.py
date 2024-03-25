import sys
import warnings
from collections import UserDict
from typing import Optional

import pyarrow as pa
import torch
import urllib3
from pyarrow.fs import FileSystem, FileType, LocalFileSystem, S3FileSystem
from torch.utils.data import Dataset, IterableDataset

from fms.datasets.util import SavableDataset


class _ArrowFileData(UserDict):
    def __init__(self, fs: FileSystem, path: str, column_name: str = "tokens"):
        self.fs = fs
        self.path = path
        self.column_name = column_name

    def __getitem__(self, idx: int):
        with self.fs.open_input_file(self.path) as file:
            reader = pa.ipc.open_file(file)
            # print(self.path, file, idx)
            return reader.get_batch(idx)[self.column_name]

    def __iter__(self):
        with self.fs.open_input_file(self.path) as file:
            reader = pa.ipc.open_file(file)
            for i in range(reader.num_record_batches):
                yield reader.get_batch(i)[self.column_name]

    def __len__(self):
        with self.fs.open_input_file(self.path) as file:
            reader = pa.ipc.open_file(file)
            return reader.num_record_batches


class ArrowFilesDataset(IterableDataset, SavableDataset):
    """
    Creates a dataset from a path to a directory of arrow files, either in
    S3/COS or a local file system.

    uri: s3://endpoint_host/path/to/files or file:///path/to/files
    world_size: for distributed training, used as the step size when stepping
        through data
    rank: for distributed training. Take every rank'th example.
    column_name: the name of the pyarrow column containing tokens
    max_seq_len: if a single record batch is longer than this, it will be split
        to avoid large memory copies. This class doesn't do packing of short
        lines, only splitting of long ones.
    """

    def __init__(
        self,
        uri: str,
        rank: int = 0,
        world_size: int = 1,
        column_name: str = "tokens",
        max_seq_len: Optional[int] = None,
    ):
        self.uri = uri
        self._rank = rank
        self.start_idx = 0
        self._step = world_size
        self.column_name = column_name
        self.max_seq_len = max_seq_len
        self.record_batch_offset = 0
        self._initialize()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._initialize()

    def state_dict(self):
        if self._rank != 0:
            warnings.warn("State dict will only be correct if taken from rank=0")
        return super().state_dict()

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
            _ArrowFileData(file_system, f.path, column_name=self.column_name)
            for f in files
            if f.type != FileType.Directory
        ]

        self._files = sorted(files, key=lambda f: f.path)

        # if not hasattr(self, "file_offset"):
        self._file_offset = self.start_idx + self._rank
        while self._file_offset >= len(self._files[0]):
            self._file_offset -= len(self._files[0])
            self._files = self._files[1:]

    def __iter__(self):
        for file in self._files:
            remainder = (len(file) - self._file_offset) % self._step
            next_file_offset = self._step - remainder

            if next_file_offset == self._step:
                next_file_offset = 0

            for i in range(self._file_offset, len(file), self._step):
                rb = file[i]
                while self.record_batch_offset < len(rb):
                    current = rb.slice(
                        offset=self.record_batch_offset, length=self.max_seq_len
                    ).to_pylist()
                    self.record_batch_offset = self.record_batch_offset + len(current)
                    t = torch.tensor(current)
                    yield t[:-1], t[1:]
                self.start_idx += self._step
                self.record_batch_offset = 0

            self._file_offset = next_file_offset


if __name__ == "__main__":
    if not len(sys.argv) == 2:
        print("Usage: python -m fms.datasets.arrow <path>")
        sys.exit(1)
    ds = ArrowFilesDataset(sys.argv[1])
    it = iter(ds)
    for i, batch in enumerate(it):
        # print(i, batch)
        if i > 100:
            break

    sd = ds.state_dict()
    print(sd)
    saved = []
    for i, batch in enumerate(it):
        # print(i, batch)
        saved.append(batch)
        if i > 5:
            break

    print("saved", saved)
    ds = ArrowFilesDataset(sys.argv[1])
    ds.load_state_dict(sd)
    print("loaded state dict", ds.state_dict())
    for i, batch in enumerate(ds):
        # print(i, batch, saved[i])
        assert saved[i] == batch
        if i > 5:
            break
