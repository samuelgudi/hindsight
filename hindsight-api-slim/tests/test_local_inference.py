"""Tests for LocalInferenceWorker, the single-thread scheduler behind the local
SentenceTransformers embeddings and reranker. The model is a plain function here."""

import asyncio
import threading
from collections.abc import Hashable

import pytest

from hindsight_api.engine.local_inference import LocalInferenceWorker


class _GatedModel:
    """Records every pass and holds the first one until released, so a test can queue
    jobs behind a pass that is still running."""

    def __init__(self) -> None:
        self.passes: list[tuple[Hashable, list[str]]] = []
        self.threads: set[int] = set()
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, key: Hashable, items: list[str]) -> list[str]:
        self.threads.add(threading.get_ident())
        self.passes.append((key, list(items)))
        if len(self.passes) == 1:
            self.started.set()
            assert self.release.wait(5)
        return [item.upper() for item in items]


async def _queue_behind_first_pass(worker: LocalInferenceWorker, model: _GatedModel) -> asyncio.Task:
    first = asyncio.create_task(worker.run(["x"]))
    assert await asyncio.to_thread(model.started.wait, 5)
    return first


async def test_every_pass_runs_on_one_thread_whatever_the_callers():
    model = _GatedModel()
    model.release.set()
    worker = LocalInferenceWorker("test", model, pass_size=8)

    results = await asyncio.gather(*(asyncio.to_thread(asyncio.run, worker.run([f"t{i}"])) for i in range(20)))

    assert results == [[f"T{i}"] for i in range(20)]
    assert len(model.threads) == 1
    assert worker._thread is not None and model.threads == {worker._thread.ident}


async def test_jobs_queued_during_a_pass_share_the_next_one_in_fifo_order():
    model = _GatedModel()
    worker = LocalInferenceWorker("test", model, pass_size=4)
    first = await _queue_behind_first_pass(worker, model)

    queued = [asyncio.create_task(worker.run([f"{n}1", f"{n}2"])) for n in "abc"]
    await asyncio.sleep(0.05)
    model.release.set()

    assert await asyncio.wait_for(asyncio.gather(first, *queued), 5) == [
        ["X"],
        ["A1", "A2"],
        ["B1", "B2"],
        ["C1", "C2"],
    ]
    assert [items for _, items in model.passes] == [["x"], ["a1", "a2", "b1", "b2"], ["c1", "c2"]]


async def test_a_job_larger_than_a_pass_is_split_and_reassembled_in_order():
    model = _GatedModel()
    model.release.set()
    worker = LocalInferenceWorker("test", model, pass_size=2)

    # Score longest first, as the embeddings do; results still come back by position.
    items = ["bb", "a", "dddd", "ccc", "e"]
    order = sorted(range(len(items)), key=lambda i: -len(items[i]))
    assert await worker.run(items, order=order) == ["BB", "A", "DDDD", "CCC", "E"]
    assert [batch for _, batch in model.passes] == [["dddd", "ccc"], ["bb", "a"], ["e"]]


async def test_a_long_item_does_not_pad_a_pass_of_short_ones():
    model = _GatedModel()
    worker = LocalInferenceWorker("test", model, pass_size=8, item_cost=len, pass_budget=12)
    first = await _queue_behind_first_pass(worker, model)

    short = asyncio.create_task(worker.run(["aa", "bb", "cc"]))
    page = asyncio.create_task(worker.run(["p" * 20]))
    tail = asyncio.create_task(worker.run(["dd", "ee"]))
    await asyncio.sleep(0.05)
    model.release.set()
    await asyncio.wait_for(asyncio.gather(first, short, page, tail), 5)

    # 3 x 2 fits the budget of 12; the 20-long page would make 4 x 20, so it waits and
    # then runs alone (a first item is always taken); the rest stays in FIFO order.
    assert [items for _, items in model.passes] == [["x"], ["aa", "bb", "cc"], ["p" * 20], ["dd", "ee"]]


async def test_different_keys_never_share_a_pass():
    model = _GatedModel()
    worker = LocalInferenceWorker("test", model, pass_size=8)
    first = await _queue_behind_first_pass(worker, model)

    queries = asyncio.create_task(worker.run(["q1"], key="query"))
    documents = asyncio.create_task(worker.run(["d1", "d2"], key="document"))
    more_queries = asyncio.create_task(worker.run(["q2"], key="query"))
    await asyncio.sleep(0.05)
    model.release.set()
    await asyncio.wait_for(asyncio.gather(first, queries, documents, more_queries), 5)

    assert model.passes[1:] == [("query", ["q1", "q2"]), ("document", ["d1", "d2"])]


async def test_a_failing_pass_fails_only_the_jobs_it_carried():
    calls: list[list[str]] = []

    def model(_key: Hashable, items: list[str]) -> list[str]:
        calls.append(items)
        if "bad" in items:
            raise RuntimeError("boom")
        return items

    worker = LocalInferenceWorker("test", model, pass_size=1)

    with pytest.raises(RuntimeError, match="boom"):
        await worker.run(["bad"])
    assert await worker.run(["good"]) == ["good"]


async def test_a_pass_returning_the_wrong_number_of_results_fails_the_job():
    worker = LocalInferenceWorker("test", lambda _key, items: items[:-1], pass_size=4)

    with pytest.raises(RuntimeError, match="returned 1 results for 2 items"):
        await worker.run(["a", "b"])


async def test_cancelled_jobs_stop_before_their_next_pass():
    model = _GatedModel()
    worker = LocalInferenceWorker("test", model, pass_size=1)
    abandoned = asyncio.create_task(worker.run(["a0", "a1", "a2"]))
    assert await asyncio.to_thread(model.started.wait, 5)
    queued = asyncio.create_task(worker.run(["queued"]))
    await asyncio.sleep(0.05)

    for task in (abandoned, queued):
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    successor = asyncio.create_task(worker.run(["successor"]))
    model.release.set()

    assert await asyncio.wait_for(successor, 5) == ["SUCCESSOR"]
    assert [items for _, items in model.passes] == [["a0"], ["successor"]]


async def test_timeout_reports_partial_results_and_counts_from_the_first_pass():
    class Budget(Exception):
        def __init__(self, partial: list[str | None]):
            self.partial = partial

    def slow(_key: Hashable, items: list[str]) -> list[str]:
        threading.Event().wait(0.06)
        return items

    worker = LocalInferenceWorker("test", slow, pass_size=2)

    with pytest.raises(Budget) as excinfo:
        await worker.run(["a", "b", "c", "d"], timeout=0.05, timeout_error=Budget)

    assert excinfo.value.partial == ["a", "b", None, None]


async def test_a_timeout_without_an_error_to_raise_is_rejected():
    worker = LocalInferenceWorker("test", lambda _key, items: items, pass_size=1)

    with pytest.raises(ValueError):
        await worker.run(["a"], timeout=1.0)


async def test_on_idle_runs_before_the_last_result_is_handed_back():
    events: list[str] = []
    worker = LocalInferenceWorker(
        "test",
        lambda _key, items: items,
        pass_size=1,
        on_idle=lambda: events.append("idle"),
    )

    await worker.run(["a", "b"])
    events.append("returned")

    assert events == ["idle", "returned"]


async def test_empty_input_does_not_start_the_thread():
    worker = LocalInferenceWorker("test", lambda _key, items: items, pass_size=1)

    assert await worker.run([]) == []
    assert worker._thread is None
