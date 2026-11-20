import queue
import threading


class IterLoader:
    """Iterable wrapper that restarts at StopIteration (labeled set is shorter).

    ``prefetch>0`` collates the next CPU batches on a daemon thread so the main
    thread can run GPU work at the same time. Sampler order is unchanged
    (shuffle=False); extra prefetched batches are dropped on ``new_epoch``.
    Thread prefetch is used instead of fork DataLoader workers because this
    code path creates CUDA tensors before the first ``iter(loader)``.
    """

    def __init__(self, loader, length=None, prefetch=0):
        self.loader = loader
        self.length = length
        self.prefetch = max(int(prefetch), 0)
        self.iter = None
        self._queue = None
        self._thread = None
        self._stop = None

    def __len__(self):
        if self.length is not None:
            return self.length
        return len(self.loader)

    def _raw_next(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            return next(self.iter)

    def new_epoch(self):
        self._stop_prefetch()
        self.iter = iter(self.loader)
        if self.prefetch > 0:
            self._start_prefetch()

    def next(self):
        if self._queue is None:
            return self._raw_next()
        item = self._queue.get()
        if isinstance(item, Exception):
            raise item
        return item

    def _start_prefetch(self):
        self._stop = threading.Event()
        self._queue = queue.Queue(maxsize=self.prefetch)
        self._thread = threading.Thread(
            target=self._prefetch_loop,
            name="iterloader-prefetch",
            daemon=True,
        )
        self._thread.start()

    def _prefetch_loop(self):
        try:
            while not self._stop.is_set():
                batch = self._raw_next()
                while not self._stop.is_set():
                    try:
                        self._queue.put(batch, timeout=0.05)
                        break
                    except queue.Full:
                        continue
        except Exception as err:
            if self._queue is not None and not self._stop.is_set():
                try:
                    self._queue.put(err, timeout=0.5)
                except Exception:
                    pass

    def _stop_prefetch(self):
        if self._thread is None:
            return
        self._stop.set()
        if self._queue is not None:
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass
        self._thread.join(timeout=5.0)
        self._thread = None
        self._queue = None
        self._stop = None
