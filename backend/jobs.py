"""Chat turns as background jobs the browser watches rather than owns.

A turn used to live inside its HTTP response: a reload dropped the stream, and with
a plain chat turn it dropped the turn itself, because the page was what recorded it.
Here every turn runs on its own thread and keeps a log of every event it emitted.
Any number of clients can subscribe to that log, from the start or from where they
left off, while it is running or after. Leaving the page never stops a job; only
`cancel` does.

One running job per chat. Jobs live in memory only — a backend restart loses the
log, not the turn, since the work records its own messages as it goes.
"""
import json
import threading
import time
import uuid

# How long a finished job stays around to be replayed or retried.
_KEEP_FINISHED_S = 3600
# How long a subscriber waits for news before checking again. Short so the thread
# behind a client that went away is handed back to the pool soon after.
_WAIT_S = 1.0


class Cancelled(Exception):
    """Raised by `Job.check` once the job has been asked to stop."""


class Busy(Exception):
    """The chat already has a running job."""


class Job:
    def __init__(self, chat_id: str | None, kind: str, rerun):
        self.id = uuid.uuid4().hex
        self.chat_id = chat_id
        self.kind = kind
        # Starts the same work again as a new job, for Retry. Holds the original
        # request, which is why the log is memory-only.
        self.rerun = rerun
        self.status = "running"
        self.error: str | None = None
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.events: list[dict] = []
        self.message_ids: list[str] = []
        self.cancel = threading.Event()
        # Set when the chat was deleted under the job: nothing more may be written
        # to it, and the client should say so rather than show an error.
        self.deleted = False
        self.thread_id: int | None = None
        self._failed_step = False
        self._cond = threading.Condition()

    def emit(self, event: dict) -> None:
        if event.get("type") == "tool_error":
            self._failed_step = True
        with self._cond:
            self.events.append(event)
            self._cond.notify_all()

    def check(self) -> None:
        if self.cancel.is_set():
            raise Cancelled()

    @property
    def running(self) -> bool:
        return self.status == "running"

    def summary(self) -> dict:
        return {"id": self.id, "chat_id": self.chat_id, "kind": self.kind,
                "status": self.status, "error": self.error,
                "started_at": self.started_at}

    def _finish(self, status: str, error: str | None = None) -> None:
        with self._cond:
            self.status = status
            self.error = error
            self.finished_at = time.time()
            if self.deleted:
                self.events.append({"type": "deleted"})
            self.events.append({"type": "end", "status": status, "error": error})
            self._cond.notify_all()

    def wait(self, timeout: float) -> bool:
        """Block until the job finishes or `timeout` passes. True if finished."""
        with self._cond:
            self._cond.wait_for(lambda: not self.running, timeout)
        return not self.running


_lock = threading.Lock()
_jobs: dict[str, Job] = {}


def _prune() -> None:
    cutoff = time.time() - _KEEP_FINISHED_S
    for jid in [j.id for j in _jobs.values()
                if j.finished_at and j.finished_at < cutoff]:
        del _jobs[jid]


def start(chat_id: str | None, kind: str, work, rerun) -> Job:
    """Run `work(job)` on a thread and return the job at once.

    `work` emits through `job.emit`, calls `job.check()` wherever stopping is safe,
    and raises on failure. Its outcome becomes the job's status: `Cancelled` (or
    anything raised after a cancel) is "cancelled", another exception is "error",
    a run with a failed tool step is "partial", anything else "done".
    """
    job = Job(chat_id, kind, rerun)
    with _lock:
        _prune()
        if chat_id and any(j.running and j.chat_id == chat_id for j in _jobs.values()):
            raise Busy()
        _jobs[job.id] = job

    def run():
        job.thread_id = threading.get_ident()
        try:
            work(job)
        except Exception as exc:
            if job.cancel.is_set():
                job._finish("cancelled")
            else:
                job.emit({"type": "error", "message": str(exc)})
                job._finish("error", str(exc))
            return
        if job.cancel.is_set():
            job._finish("cancelled")
        else:
            job._finish("partial" if job._failed_step else "done")

    job.emit({"type": "job", "id": job.id, "kind": kind})
    threading.Thread(target=run, daemon=True).start()
    return job


def get(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def latest_for_chat(chat_id: str) -> Job | None:
    with _lock:
        mine = [j for j in _jobs.values() if j.chat_id == chat_id and not j.deleted]
    return max(mine, key=lambda j: j.started_at) if mine else None


def running() -> list[Job]:
    with _lock:
        return [j for j in _jobs.values() if j.running and not j.deleted]


def subscribe(job: Job, after: int = 0):
    """NDJSON lines for `job`'s events from index `after`: the log, then live."""
    i = after
    while True:
        with job._cond:
            job._cond.wait_for(lambda: len(job.events) > i or not job.running, _WAIT_S)
            batch = job.events[i:]
            done = not job.running
        for event in batch:
            yield json.dumps(event) + "\n"
        i += len(batch)
        if done and i >= len(job.events):
            return
