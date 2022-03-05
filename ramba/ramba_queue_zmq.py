"""
Copyright 2021 Intel Corporation

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# This version provides a ZeroMQ version of ramba_queue
import asyncio
import threading
from typing import Optional, Any, List, Dict
from collections.abc import Iterable

# import cloudpickle as cp
# import pyarrow
import pickle as pick

if pick.HIGHEST_PROTOCOL < 5:
    import pickle5 as pick

import zmq

context = zmq.Context()
sockets = {}
prefiltered = {}

import time

import socket


def get_my_ip(hint_ip=None):
    hint_ip = "172.13.1.1" if hint_ip is None else hint_ip
    # hint_ip = "192.168.8.1" if hint_ip is None else hint_ip
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((hint_ip, 1))
    return s.getsockname()[0]


class Empty(Exception):
    pass


class Full(Exception):
    pass


def pickle(item: Any) -> Any:
    # return cp.dumps(item)
    # return pyarrow.serialize(item).to_buffer()
    # return pick.dumps(item,protocol=5)
    rv = [0]
    rv0 = pick.dumps(item, protocol=5, buffer_callback=rv.append)
    rv[0] = rv0
    return rv


def unpickle(msg0):
    # msg = cp.loads(msg0)
    # msg = pick.loads(msg0)
    msg = pick.loads(msg0[0], buffers=msg0[1:])
    # msg = pyarrow.deserialize(msg0)
    return msg


send_list = []
send_lock = threading.Lock()

# NOTE: this still has a race condition with threads;  
# but it should be good enough to fix reentrant case resulting from deletion of arrays
def real_send(s, msg):
    send_list.append((s, msg))
    if (send_lock.acquire(blocking=False)):
        while send_list:
            (s,msg) = send_list.pop(0)
            s.send_multipart(msg, copy=False)
        send_lock.release()


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

    # tag is ignored -- for compatibility with mpi version
    def __init__(self, maxsize: int = 0, hint_ip=None, tag=0) -> None:
        self.maxsize = maxsize
        self.ip = get_my_ip(hint_ip)
        s = context.socket(zmq.PULL)
        self.port = s.bind_to_random_port("tcp://" + self.ip)
        sockets[self] = s
        self.address = "tcp://" + self.ip + ":" + str(self.port)
        prefiltered[self] = []
        self.sent_data = 0
        self.recv_data = 0
        self.pickle_time = 0.0
        self.unpickle_time = 0.0
        # print ("Created Queue ADDR:",self.ip,self.port, "HINT:",hint_ip)

    def get_stats(self):
        return (
            self.ip,
            self.port,
            self.recv_data,
            self.sent_data,
            self.unpickle_time,
            self.pickle_time,
        )

    '''
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
    '''

    def put(
        self,
        item: Any,
        block: bool = True,
        timeout: Optional[float] = None,
        raw: Optional[bool] = False,
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
        """
        if self not in sockets:
            s = context.socket(zmq.PUSH)
            s.connect(self.address)
            sockets[self] = s
        else:
            s = sockets[self]
        t = -time.time()
        msg = item if raw else pickle(item)
        t += time.time()
        # s.send(msg,copy=False)
        # s.send_multipart(msg, copy=False)
        real_send(s, msg)
        # s.send_multipart(msg)
        # self.sent_data+=len(msg)
        # self.sent_data+=len(msg[0])
        for d in msg:
            self.sent_data += (
                len(d.raw()) if isinstance(d, pick.PickleBuffer) else len(d)
            )
        if not raw:
            self.pickle_time += t

    '''
    async def put_async(self,
                        item: Any,
                        block: bool = True,
                        timeout: Optional[float] = None) -> None:
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
    '''

    def get(
        self,
        block: bool = True,
        gfilter=lambda x: True,
        timeout: Optional[float] = None,
        print_times: Optional[bool] = False,
        raw: Optional[bool] = False,
        msginfo=lambda x: "",
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
        t0 = time.time()
        pf = prefiltered[self]
        s = sockets[self]
        if not raw:
            for i in range(len(pf)):
                if gfilter(pf[i]):
                    msg = pf[i]
                    del pf[i]
                    t1 = time.time()
                    if print_times:
                        print(
                            "Get msg:  from prefiltered ",
                            (t1 - t0) * 1000,
                            msginfo(msg),
                        )
                    return msg
        while True:
            # msg0 = s.recv()
            msg0 = s.recv_multipart(copy=False)
            # success=False
            # while not success:
            #    try:
            #        msg0 = s.recv_multipart(copy=False,flags=zmq.NOBLOCK)
            #        success=True
            #    except:
            #        sucess=False
            # self.recv_data+=len(msg0)
            for d in msg0:
                self.recv_data += len(d)
            t1 = time.time()
            if not raw:
                msg = unpickle(msg0)
                t2 = time.time()
                self.unpickle_time += t2 - t1
                if gfilter(msg):
                    # if print_times: print("Get msg: from queue ", len(msg0),"bytes",(t1-t0)*1000, "ms,  unpickle ", (t2-t1)*1000,"ms")
                    if print_times:
                        print(
                            "Get msg: from queue ",
                            sum(
                                [
                                    len(i.raw())
                                    if isinstance(i, pick.PickleBuffer)
                                    else len(i)
                                    for i in msg0
                                ]
                            ),
                            "bytes",
                            (t1 - t0) * 1000,
                            "ms,  unpickle ",
                            (t2 - t1) * 1000,
                            "ms",
                            msginfo(msg),
                        )
                    return msg
                else:
                    pf.append(msg)
            else:
                if print_times:
                    print(
                        "Get msg: from queue ",
                        sum(
                            [
                                len(i.raw())
                                if isinstance(i, pick.PickleBuffer)
                                else len(i)
                                for i in msg0
                            ]
                        ),
                        "bytes",
                        (t1 - t0) * 1000,
                        "ms,  raw message",
                    )
                return msg0

        """
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
        """

    def multi_get(
        self,
        n: int = 1,
        gfilter=lambda x: True,
        timeout: Optional[float] = None,
        print_times=False,
        msginfo=lambda x: "",
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
        return [
            self.get(gfilter=gfilter, print_times=print_times, msginfo=msginfo)
            for _ in range(n)
        ]


'''
    async def get_async(self,
                        block: bool = True,
                        gfilter = lambda x: True,
                        timeout: Optional[float] = None) -> Any:
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
        while n>0:
            try:
                m = await asyncio.wait_for(self.queue.get(), timeout)
                if filt(m): n-=1
                res.append(m)
            except asyncio.TimeoutError:
                break
        return res

    def put_nowait(self, item):
        self.queue.put_nowait(item)

    def put_nowait_batch(self, items):
        # If maxsize is 0, queue is unbounded, so no need to check size.
        if self.maxsize > 0 and len(items) + self.qsize() > self.maxsize:
            raise Full(f"Cannot add {len(items)} items to queue of size "
                       f"{self.qsize()} and maxsize {self.maxsize}.")
        for item in items:
            self.queue.put_nowait(item)

    def get_nowait(self):
        return self.queue.get_nowait()

    def get_nowait_batch(self, num_items):
        if num_items > self.qsize():
            raise Empty(f"Cannot get {num_items} items from queue of size "
                        f"{self.qsize()}.")
        return [self.queue.get_nowait() for _ in range(num_items)]
'''
