"""Tests for map_local_model_weights_read_only: CPU weights re-pointed at a read-only
mapping of their safetensors file, with misaligned files copied once with a padded header."""

import json
import struct

import pytest

torch = pytest.importorskip("torch")

from hindsight_api.engine import local_device  # noqa: E402
from hindsight_api.engine.local_device import (  # noqa: E402
    _READ_ONLY_WEIGHT_MAPPINGS,
    map_local_model_weights_read_only,
)


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(50, 8)
        self.proj = torch.nn.Linear(8, 4)

    def forward(self, ids):
        return self.proj(self.embed(ids))


def _write_safetensors(path, tensors: dict, data_start_mod8: int) -> None:
    """Write a safetensors file by hand so the data section's alignment is chosen."""
    header: dict = {}
    blobs = []
    offset = 0
    for name, tensor in tensors.items():
        raw = tensor.detach().contiguous().numpy().tobytes()
        header[name] = {"dtype": "F32", "shape": list(tensor.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    encoded += b" " * ((data_start_mod8 - (8 + len(encoded))) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for raw in blobs:
            f.write(raw)


@pytest.fixture
def weights_dir(tmp_path, monkeypatch):
    from huggingface_hub import constants

    # Aligned copies go next to the HF cache; keep them inside the test's tmp dir.
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    return snapshot


@pytest.mark.parametrize("data_start_mod8", [0, 2])
def test_weights_end_up_on_a_read_only_mapping_with_identical_output(weights_dir, data_start_mod8):
    torch.manual_seed(0)
    reference = _Tiny()
    path = weights_dir / "model.safetensors"
    _write_safetensors(path, dict(reference.state_dict()), data_start_mod8)

    model = _Tiny()
    model.load_state_dict(reference.state_dict())
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        before = model(ids)

    # The file says "embed.weight" while the wrapper's parameter is "model.embed.weight",
    # the way a SentenceTransformer names its transformer's weights "0.auto_model.<key>".
    wrapper = torch.nn.Module()
    wrapper.model = model
    count_before = len(_READ_ONLY_WEIGHT_MAPPINGS)
    mapped = map_local_model_weights_read_only(wrapper, str(path), label="test")

    expected = sum(t.numel() * t.element_size() for t in reference.state_dict().values())
    assert mapped == expected
    assert len(_READ_ONLY_WEIGHT_MAPPINGS) == count_before + 1
    mapping = _READ_ONLY_WEIGHT_MAPPINGS[-1]
    with torch.inference_mode():
        assert torch.equal(model(ids), before)
    for tensor in model.state_dict().values():
        assert tensor.data_ptr() % tensor.element_size() == 0
    assert mapping.size() >= expected
    if data_start_mod8:
        copies = list((weights_dir.parent.parent / "hindsight-aligned-weights").iterdir())
        assert [c.name for c in copies] == ["abc123-model.safetensors"]
        with open(copies[0], "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
        assert (8 + header_len) % 8 == 0


def test_tensors_that_do_not_match_the_file_are_left_alone(weights_dir):
    reference = _Tiny()
    path = weights_dir / "model.safetensors"
    state = reference.state_dict()
    # Wrong shape for one tensor, and one tensor missing from the file.
    _write_safetensors(path, {"embed.weight": torch.zeros(10, 8), "proj.weight": state["proj.weight"]}, 0)

    model = _Tiny()
    original_embed = model.embed.weight.data_ptr()
    mapped = map_local_model_weights_read_only(model, str(path), label="test")

    assert mapped == state["proj.weight"].numel() * 4
    assert model.embed.weight.data_ptr() == original_embed


def test_a_file_with_no_matching_tensor_keeps_no_mapping(weights_dir):
    path = weights_dir / "model.safetensors"
    _write_safetensors(path, {"other.weight": torch.zeros(3, 3)}, 0)
    count_before = len(_READ_ONLY_WEIGHT_MAPPINGS)

    assert map_local_model_weights_read_only(_Tiny(), str(path), label="test") == 0
    assert len(_READ_ONLY_WEIGHT_MAPPINGS) == count_before


def test_resolve_local_safetensors_finds_a_local_directory(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    assert local_device.resolve_local_safetensors(str(tmp_path)) == str(tmp_path / "model.safetensors")
    assert local_device.resolve_local_safetensors(str(tmp_path / "missing")) is None
