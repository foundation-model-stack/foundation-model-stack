import os.path
import tempfile

import pytest

from fms.distributed.strategy import NotDistributed, NoOpStrategy
from fms.models.llama import LLaMA


@pytest.fixture
def text_file_path(tmpdir):
    f = open(f"{tmpdir}/file.txt", "w+")
    f.write("some line")
    f.close()
    yield f"{tmpdir}/file.txt"


@pytest.fixture(params=[NotDistributed, NoOpStrategy])
def mock_llama2():
    return LLaMA(
        src_vocab_size=256,
        emb_dim=16,
        multiple_of=2,
        nheads=2,
        nlayers=2,
        norm_eps=1e-05,
        pad_id=0,
    )


@pytest.fixture
def model_location(tmpdir, mock_llama2):
    mock_llama2.save(f"{tmpdir}/saved_model")
    yield f"{tmpdir}/saved_model"


def _assert_all_model_files_exist(model_path):
    assert os.path.exists(model_path)
    assert os.path.exists(f"{model_path}/config.json") and os.path.isfile(
        f"{model_path}/config.json"
    )
    assert os.path.exists(f"{model_path}/model_state.pth") and os.path.isfile(
        f"{model_path}/model_state.pth"
    )
    assert os.path.exists(f"{model_path}/distributed_strategy.bin") and os.path.isfile(
        f"{model_path}/distributed_strategy.bin"
    )


def test_save(mock_llama2):
    with tempfile.TemporaryDirectory() as workdir:
        mock_llama2.save(f"{workdir}/llama_model")
        _assert_all_model_files_exist(f"{workdir}/llama_model")


def test_save_dir_already_exists(mock_llama2):
    with tempfile.TemporaryDirectory() as workdir:
        os.mkdir(f"{workdir}/llama_model")
        mock_llama2.save(f"{workdir}/llama_model")
        _assert_all_model_files_exist(f"{workdir}/llama_model")


def test_load(model_location, mock_llama2):
    llama = LLaMA.load(model_location)
    assert llama.get_config().as_dict() == mock_llama2.get_config().as_dict()

    m1_sd = llama.state_dict()
    m2_sd = mock_llama2.state_dict()
    assert m1_sd.keys() == m2_sd.keys()
    is_equal_weights = True
    for k in m1_sd.keys():
        if m1_sd[k].ne(m2_sd[k]).sum() > 0:
            is_equal_weights = False
            break
    assert is_equal_weights
    assert type(llama.distributed_strategy) == type(mock_llama2.distributed_strategy)


def test_save_not_directory(mock_llama2, text_file_path):
    with pytest.raises(NotADirectoryError):
        mock_llama2.save(text_file_path)


def test_load_file_not_found(tmpdir):
    with pytest.raises(FileNotFoundError):
        LLaMA.load(f"{tmpdir}/no_file")


def test_load_not_directory(text_file_path):
    with pytest.raises(NotADirectoryError):
        LLaMA.load(text_file_path)
