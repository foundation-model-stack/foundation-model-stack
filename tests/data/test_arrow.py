import os
import random
import tempfile

import pyarrow as pa
import torch

from fms.datasets.arrow import ArrowFilesDataset


# Generates test data in `<mock_data_path>`.
# arrow_partitions = number of arrow files to create
# batches_per_partition is the number of "documents" in each arrow file.
# Across the full dataset of `N=arrow_partitions * batches_per_partition`
# batches, first token in each batch begins with its batch idx.
def generate_test_data(
    mock_dataset_path, arrow_partitions, batches_per_partition, max_tok_per_batch=50
):
    # the rows of tokens in each batch start with a row number to ease testing
    next_token = 0
    random.seed(42)
    schema = pa.schema([("tokens", pa.uint32())])
    os.makedirs(mock_dataset_path, exist_ok=True)
    if not mock_dataset_path.endswith("/"):
        mock_dataset_path = mock_dataset_path + "/"

    for i in range(arrow_partitions):
        file = mock_dataset_path + f"part-{i:05d}.arrow"
        print("generating mock data in", file)
        with pa.RecordBatchFileWriter(file, schema) as writer:
            for _ in range(batches_per_partition):
                tokens_in_batch = random.randint(5, max_tok_per_batch)
                tokens = [next_token]
                for _ in range(tokens_in_batch):
                    tokens.append(random.randint(0, 1000))
                batch = pa.RecordBatch.from_arrays([pa.array(tokens)], schema=schema)
                writer.write_batch(batch)
                next_token += 1


def test_arrow_dataset_reload():
    """
    Validates that the dataset can be reloaded from a saved state dict.
    """
    with tempfile.TemporaryDirectory() as data_path:
        generate_test_data(data_path, 8, 97)
        data_path = "file:///" + data_path

        ds = ArrowFilesDataset(data_path)
        it = iter(ds)
        for i, batch in enumerate(it):
            assert batch[0][0] == i
            if i > 100:
                break

        sd = ds.state_dict()
        # print(sd)
        # save some data from after where it was saved, so we can verify
        # that it gets the same data after loading the sd
        saved = []
        for i, batch in enumerate(it):
            # print(i, batch)
            saved.append(batch[0])
            if i > 5:
                break

        # print("saved", saved)
        ds = ArrowFilesDataset(data_path)
        ds.load_state_dict(sd)
        # print("loaded state dict", ds.state_dict())
        for i, batch in enumerate(ds):
            torch.testing.assert_allclose(saved[i], batch[0])
            if i > 5:
                break


def test_ranked():
    """
    test handling of rank and world size in saved state dict.
    """
    with tempfile.TemporaryDirectory() as data_path:
        generate_test_data(data_path, 7, 95)
        data_path = "file:///" + data_path

        rank = 7
        world = 13

        # we actually save the sd from rank 0. other ranks take every
        # rank'th row after the zeroth row.
        ds0 = ArrowFilesDataset(data_path, 0, world)
        ds = ArrowFilesDataset(data_path, rank, world)

        it = iter(ds)
        iter0 = iter(ds0)
        for i, batch in enumerate(it):
            next(iter0)
            v = batch[0][0]
            assert v % world == rank
            if v > 250:
                break

        sd = ds0.state_dict()
        print(sd)
        saved = []
        for i, batch in enumerate(it):
            saved.append(batch[0])
            if i > 5:
                break

        ds = ArrowFilesDataset(data_path, rank, world)
        ds.load_state_dict(sd)
        for i, batch in enumerate(ds):
            # print(i, batch, saved[i])
            torch.testing.assert_allclose(saved[i], batch[0])
            if i > 5:
                break


def test_split():
    """
    in cases where a row is longer than max seq len, it's chunked and
    read as chunks. this test checks the handling of the state dict for
    picking up after the last chunk.
    """
    with tempfile.TemporaryDirectory() as data_path:
        generate_test_data(data_path, 7, 10)
        data_path = "file:///" + data_path

        max_seq_len = 13
        ds = ArrowFilesDataset(data_path, max_seq_len=max_seq_len)

        it = iter(ds)
        for i, batch in enumerate(it):
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

        # print("saved", saved)
        ds = ArrowFilesDataset(data_path, max_seq_len=max_seq_len)
        ds.load_state_dict(sd)
        # print("loaded state dict", ds.state_dict())
        for i, batch in enumerate(ds):
            torch.testing.assert_allclose(saved[i][0], batch[0])
            torch.testing.assert_allclose(saved[i][1], batch[1])
            if i > 5:
                break
