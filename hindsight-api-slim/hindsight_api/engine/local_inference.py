"""One worker thread per in-process model, with concurrent requests packed into shared passes.

Local embeddings used to run on asyncio's default executor (``asyncio.to_thread``) and the
local reranker on its own pool of ``RERANKER_LOCAL_MAX_CONCURRENT`` threads. Every distinct
thread that runs a torch forward pass gets its own OpenMP team (``OMP_NUM_THREADS`` workers)
and its own MKL scratch buffers, and MKL keeps those buffers for as long as the thread
lives. The default executor grows to ``min(32, cpu_count + 4)`` threads, so a busy daemon
ended up with hundreds of native threads and several GB of committed memory that never came
back: measured on a 16-thread Windows host, 329 threads and commit going from 5 to 8.4 GB in
the first 15 minutes, 11.5 GB after a few days. Windows has no overcommit, so that commit is
charged against RAM + pagefile even though most of it is never touched. The extra threads do
not buy throughput either: N passes at once on N x OMP threads fight over the same cores.

So each model gets exactly one thread that owns it, and callers queue for it. The worker
packs items from every waiting job into passes of at most ``pass_size`` items, FIFO by job:
a one-text recall query that arrives behind a 200-text retain rides the next pass instead of
waiting for the whole retain to finish. Nothing waits on a timer. An idle worker runs a lone
job at once, and batches form only out of what queued up while the previous pass ran
(continuous batching, the way TEI and vLLM schedule).

The queue is threading-based and the result is a ``concurrent.futures.Future``, so a worker
can be shared by every event loop in the process; each caller awaits it with
``asyncio.wrap_future`` on its own loop.
"""

import asyncio
import logging
import queue
import threading
import time
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


@dataclass(eq=False)
class _Job(Generic[ItemT, ResultT]):
    """One caller's items, how far the worker got through them, and where the result goes."""

    key: Hashable
    items: list[ItemT]
    # Positions of ``items`` in the order they should be scored (e.g. sorted by length to
    # cut padding). Results are always written back to their original position.
    order: list[int]
    future: Future
    timeout: float
    timeout_error: Callable[[list[ResultT | None]], BaseException] | None
    results: list[ResultT | None] = field(init=False)
    cursor: int = 0
    # Set when the job's first pass starts, so the budget measures time spent scoring,
    # not time spent queued behind other jobs (the same clock the per-call loop used).
    deadline: float | None = None
    # Set by the caller when its task is cancelled. Needed because once the worker has
    # started a job, Future.cancel() can no longer stop it.
    cancelled: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self.results = [None] * len(self.items)

    @property
    def done(self) -> bool:
        return self.cursor >= len(self.order)


@dataclass
class _Slice:
    """The part of one job that a pass carries: which of its positions, in pass order."""

    job: _Job
    positions: list[int]


class LocalInferenceWorker(Generic[ItemT, ResultT]):
    """Serialises one model's forward passes on a single thread and packs callers together.

    ``run_pass(key, items)`` does one forward pass and returns one result per item, in
    order. Items from different jobs share a pass only when their ``key`` is equal (the
    embeddings use it to keep query and document prompts apart). ``on_idle`` runs on the
    worker thread whenever it has no work left, before the last results are handed back,
    which is where per-batch memory release belongs.
    """

    def __init__(
        self,
        name: str,
        run_pass: Callable[[Hashable, list[ItemT]], Sequence[ResultT]],
        pass_size: int,
        on_idle: Callable[[], None] | None = None,
    ):
        self.name = name
        self._run_pass = run_pass
        self.pass_size = max(1, pass_size)
        self._on_idle = on_idle
        self._inbox: queue.SimpleQueue[_Job] = queue.SimpleQueue()
        self._start_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    async def run(
        self,
        items: list[ItemT],
        *,
        key: Hashable = None,
        order: list[int] | None = None,
        timeout: float = 0.0,
        timeout_error: Callable[[list[ResultT | None]], BaseException] | None = None,
    ) -> list[ResultT]:
        """Queue ``items`` for the model and wait for one result per item.

        ``timeout`` > 0 bounds the time spent scoring this job; when it runs out the job
        fails with ``timeout_error(partial_results)``, where unscored positions are None.
        Cancelling the caller drops whatever part of the job has not reached a pass yet.
        """
        if not items:
            return []
        if timeout > 0 and timeout_error is None:
            raise ValueError("a timeout needs a timeout_error to report the partial results with")
        job: _Job = _Job(
            key=key,
            items=items,
            order=order if order is not None else list(range(len(items))),
            future=Future(),
            timeout=timeout,
            timeout_error=timeout_error,
        )
        self._submit(job)
        try:
            return await asyncio.wrap_future(job.future)
        except asyncio.CancelledError:
            job.cancelled.set()
            raise

    def _submit(self, job: _Job) -> None:
        self._inbox.put(job)
        # The thread is started on first use rather than in __init__: tests and the
        # providers' own __init__ build these objects without ever running a model.
        with self._start_lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name=self.name, daemon=True)
                self._thread.start()

    def _admit(self, job: _Job, active: list[_Job]) -> None:
        # False means the caller cancelled while the job was still queued: drop it unseen.
        if job.future.set_running_or_notify_cancel():
            active.append(job)

    def _loop(self) -> None:
        active: list[_Job] = []
        while True:
            if not active:
                self._admit(self._inbox.get(), active)
            while True:
                try:
                    self._admit(self._inbox.get_nowait(), active)
                except queue.Empty:
                    break
            active = self._drop_cancelled_and_expired(active)
            if not active:
                continue
            slices = self._pack(active)
            batch = [s.job.items[i] for s in slices for i in s.positions]
            started = time.monotonic()
            for s in slices:
                if s.job.deadline is None and s.job.timeout > 0:
                    s.job.deadline = started + s.job.timeout
            try:
                outputs = list(self._run_pass(slices[0].job.key, batch))
                if len(outputs) != len(batch):
                    raise RuntimeError(f"{self.name}: pass returned {len(outputs)} results for {len(batch)} items")
            except Exception as exc:
                failed = {id(s.job) for s in slices}
                active = [job for job in active if id(job) not in failed]
                self._idle_if_drained(active)
                for s in slices:
                    s.job.future.set_exception(exc)
                continue
            offset = 0
            for s in slices:
                for position in s.positions:
                    s.job.results[position] = outputs[offset]
                    offset += 1
                s.job.cursor += len(s.positions)
            finished = [job for job in active if job.done]
            active = [job for job in active if not job.done]
            self._idle_if_drained(active)
            for job in finished:
                job.future.set_result(job.results)

    def _drop_cancelled_and_expired(self, active: list[_Job]) -> list[_Job]:
        now = time.monotonic()
        kept: list[_Job] = []
        for job in active:
            if job.cancelled.is_set():
                # Nobody awaits this future any more; resolve it so it does not linger.
                job.future.set_exception(asyncio.CancelledError())
            elif job.deadline is not None and job.timeout_error is not None and now >= job.deadline:
                job.future.set_exception(job.timeout_error(job.results))
            else:
                kept.append(job)
        return kept

    def _pack(self, active: list[_Job]) -> list[_Slice]:
        """Fill one pass from the oldest job onward, skipping jobs keyed differently."""
        key = active[0].key
        room = self.pass_size
        slices: list[_Slice] = []
        for job in active:
            if room == 0:
                break
            if job.key != key:
                continue
            take = job.order[job.cursor : job.cursor + room]
            slices.append(_Slice(job=job, positions=take))
            room -= len(take)
        return slices

    def _idle_if_drained(self, active: list[_Job]) -> None:
        if active or not self._inbox.empty() or self._on_idle is None:
            return
        try:
            self._on_idle()
        except Exception:  # pragma: no cover - defensive: never kill the worker over cleanup
            logger.exception("%s: on_idle hook failed", self.name)
