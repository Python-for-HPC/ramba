"""
Copyright 2021 Intel Corporation

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import asyncio
from typing import Optional, Any, List, Dict
from collections.abc import Iterable

import ray


class Empty(Exception):
    pass


class Full(Exception):
    pass


class Queue:
    """A first-in, first-out queue implementation on Ray.

    The behavior and use cases are similar to those of the asyncio.Queue class.

    Features both sync and async put and get methods.  Provides the option to
    block until space is available when calling put on a full queue,
    or to block until items are available when calling get on an empty queue.

    Optionally supports batched put and get operations to minimize
    serialization overhead.

    Args:
        maxsize (optional, int): maximum size of the queue. If zero, size is
            unbounded.
        actor_options (optional, Dict): Dictionary of options to pass into
            the QueueActor during creation. These are directly passed into
            QueueActor.options(...). This could be useful if you
            need to pass in custom resource requirements, for example.

    Examples:
        >>> q = Queue()
        >>> items = list(range(10))
        >>> for item in items:
        >>>     q.put(item)
        >>> for item in items:
        >>>     assert item == q.get()
        >>> # Create Queue with the underlying actor reserving 1 CPU.
        >>> q = Queue(actor_options={"num_cpus": 1})
    """

    def __init__(
        self,
        maxsize: int = 0,
        hint_ip=None,
        tag=0,
        actor_options: Optional[Dict] = None,
    ) -> None:
        actor_options = actor_options or {}
        self.maxsize = maxsize
        self.actor = (
            ray.remote(_QueueActor).options(**actor_options).remote(self.maxsize)
        )
        self.prefiltered = []

    def __len__(self) -> int:
        return self.size()

    def size(self) -> int:
        """The size of the queue."""
        return ray.get(self.actor.qsize.remote())

    def qsize(self) -> int:
        """The size of the queue."""
        return self.size()

    def empty(self) -> bool:
        """Whether the queue is empty."""
        return ray.get(self.actor.empty.remote())

    def full(self) -> bool:
        """Whether the queue is full."""
        return ray.get(self.actor.full.remote())

    def put(
        self, item: Any, block: bool = True, timeout: Optional[float] = None
    ) -> None:
        """Adds an item to the queue.

        If block is True and the queue is full, blocks until the queue is no
        longer full or until timeout.

        There is no guarantee of order if multiple producers put to the same
        full queue.

        Raises:
            Full: if the queue is full and blocking is False.
            Full: if the queue is full, blocking is True, and it timed out.
            ValueError: if timeout is negative.
        """
        if not block:
            try:
                ray.get(self.actor.put_nowait.remote(item))
            except asyncio.QueueFull:
                raise Full
        else:
            if timeout is not None and timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                ray.get(self.actor.put.remote(item, timeout))

    async def put_async(
        self, item: Any, block: bool = True, timeout: Optional[float] = None
    ) -> None:
        """Adds an item to the queue.

        If block is True and the queue is full,
        blocks until the queue is no longer full or until timeout.

        There is no guarantee of order if multiple producers put to the same
        full queue.

        Raises:
            Full: if the queue is full and blocking is False.
            Full: if the queue is full, blocking is True, and it timed out.
            ValueError: if timeout is negative.
        """
        if not block:
            try:
                await self.actor.put_nowait.remote(item)
            except asyncio.QueueFull:
                raise Full
        else:
            if timeout is not None and timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                await self.actor.put.remote(item, timeout)

    def get(
        self,
        block: bool = True,
        gfilter=lambda x: True,
        timeout: Optional[float] = None,
    ) -> Any:
        """Gets an item from the queue.

        If block is True and the queue is empty, blocks until the queue is no
        longer empty or until timeout.

        There is no guarantee of order if multiple consumers get from the
        same empty queue.

        Returns:
            The next item in the queue.

        Raises:
            Empty: if the queue is empty and blocking is False.
            Empty: if the queue is empty, blocking is True, and it timed out.
            ValueError: if timeout is negative.
        """
        for i in range(len(self.prefiltered)):
            if gfilter(self.prefiltered[i]):
                msg = self.prefiltered[i]
                del self.prefiltered[i]
                return msg

        if not block:
            try:
                msg = ray.get(self.actor.get_nowait.remote())
                if gfilter(msg):
                    return msg
                else:
                    self.prefiltered.append(msg)
            except asyncio.QueueEmpty:
                raise Empty
        else:
            # FIXME: Timeout needs to be fixed here!
            while True:
                if timeout is not None and timeout < 0:
                    raise ValueError("'timeout' must be a non-negative number")
                else:
                    msg = ray.get(self.actor.get.remote(timeout))
                    if gfilter(msg):
                        return msg
                    else:
                        self.prefiltered.append(msg)

    def multi_get(
        self, n: int = 1, gfilter=lambda x: True, timeout: Optional[float] = None
    ) -> Any:
        """Get multiple items from the queue.
        Blocking call.  
        There is no guarantee of order if multiple consumers get from the
        same empty queue.

        Returns:
            List of next n items matching predicate gfilter, or fewer in case of timeout.

        Raises:
            ValueError: if timeout or n is negative.
        """
        if timeout is not None and timeout < 0:
            raise ValueError("'timeout' must be a non-negative number")
        if n < 0:
            raise ValueError("'n' must be a non-negative number")
        res = []

        if n > 0:
            j = 0
            for i in range(len(self.prefiltered)):
                if gfilter(self.prefiltered[i - j]):
                    res.append(self.prefiltered[i - j])
                    del self.prefiltered[i - j]
                    j += 1
                    n -= 1
                    if n == 0:
                        break

        if n > 0:
            msgs = ray.get(self.actor.mf_get.remote(n, gfilter, timeout))
            for m in msgs:
                if gfilter(m):
                    res.append(m)
                else:
                    self.prefiltered.append(m)

        return res

    async def get_async(
        self,
        block: bool = True,
        gfilter=lambda x: True,
        timeout: Optional[float] = None,
    ) -> Any:
        """Gets an item from the queue.

        There is no guarantee of order if multiple consumers get from the
        same empty queue.

        Returns:
            The next item in the queue.
        Raises:
            Empty: if the queue is empty and blocking is False.
            Empty: if the queue is empty, blocking is True, and it timed out.
            ValueError: if timeout is negative.
        """
        if not block:
            try:
                return await self.actor.get_nowait.remote()
            except asyncio.QueueEmpty:
                raise Empty
        else:
            if timeout is not None and timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                return await self.actor.get.remote(timeout)

    def put_nowait(self, item: Any) -> None:
        """Equivalent to put(item, block=False).

        Raises:
            Full: if the queue is full.
        """
        return self.put(item, block=False)

    def put_nowait_batch(self, items: Iterable) -> None:
        """Takes in a list of items and puts them into the queue in order.

        Raises:
            Full: if the items will not fit in the queue
        """
        if not isinstance(items, Iterable):
            raise TypeError("Argument 'items' must be an Iterable")

        ray.get(self.actor.put_nowait_batch.remote(items))

    def get_nowait(self) -> Any:
        """Equivalent to get(block=False).

        Raises:
            Empty: if the queue is empty.
        """
        return self.get(block=False)

    def get_nowait_batch(self, num_items: int) -> List[Any]:
        """Gets items from the queue and returns them in a
        list in order.

        Raises:
            Empty: if the queue does not contain the desired number of items
        """
        if not isinstance(num_items, int):
            raise TypeError("Argument 'num_items' must be an int")
        if num_items < 0:
            raise ValueError("'num_items' must be nonnegative")

        return ray.get(self.actor.get_nowait_batch.remote(num_items))

    def shutdown(self, force: bool = False, grace_period_s: int = 5) -> None:
        """Terminates the underlying QueueActor.

        All of the resources reserved by the queue will be released.

        Args:
            force (bool): If True, forcefully kill the actor, causing an
                immediate failure. If False, graceful
                actor termination will be attempted first, before falling back
                to a forceful kill.
            grace_period_s (int): If force is False, how long in seconds to
                wait for graceful termination before falling back to
                forceful kill.
        """
        if self.actor:
            if force:
                ray.kill(self.actor, no_restart=True)
            else:
                done_ref = self.actor.__ray_terminate__.remote()
                done, not_done = ray.wait([done_ref], timeout=grace_period_s)
                if not_done:
                    ray.kill(self.actor, no_restart=True)
        self.actor = None


class _QueueActor:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.queue = asyncio.Queue(self.maxsize)

    def qsize(self):
        return self.queue.qsize()

    def empty(self):
        return self.queue.empty()

    def full(self):
        return self.queue.full()

    async def put(self, item, timeout=None):
        try:
            await asyncio.wait_for(self.queue.put(item), timeout)
        except asyncio.TimeoutError:
            raise Full

    async def get(self, timeout=None):
        try:
            return await asyncio.wait_for(self.queue.get(), timeout)
        except asyncio.TimeoutError:
            raise Empty

    # Returns all messages up to and including the nth one matching condition filt
    async def mf_get(self, n, filt=lambda x: True, timeout=None):
        res = []
        while n > 0:
            try:
                m = await asyncio.wait_for(self.queue.get(), timeout)
                if filt(m):
                    n -= 1
                res.append(m)
            except asyncio.TimeoutError:
                break
        return res

    def put_nowait(self, item):
        self.queue.put_nowait(item)

    def put_nowait_batch(self, items):
        # If maxsize is 0, queue is unbounded, so no need to check size.
        if self.maxsize > 0 and len(items) + self.qsize() > self.maxsize:
            raise Full(
                f"Cannot add {len(items)} items to queue of size "
                f"{self.qsize()} and maxsize {self.maxsize}."
            )
        for item in items:
            self.queue.put_nowait(item)

    def get_nowait(self):
        return self.queue.get_nowait()

    def get_nowait_batch(self, num_items):
        if num_items > self.qsize():
            raise Empty(
                f"Cannot get {num_items} items from queue of size " f"{self.qsize()}."
            )
        return [self.queue.get_nowait() for _ in range(num_items)]
