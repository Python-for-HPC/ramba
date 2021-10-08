"""
Copyright 2021 Intel Corporation

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# This version provides a MPI version of ramba_queue
from typing import Optional, Any, List, Dict
import time
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
MPI.Attach_buffer(np.empty(1000000,dtype=np.int32))
prefiltered = {}

class Queue:
    """A first-in, first-out queue implementation on Ray.

    The behavior and use cases are similar to those of the asyncio.Queue class.

    """

    def __init__(self, maxsize: int = 0, hint_ip=None, tag=0) -> None:
        self.tag = tag
        self.rank = rank
        prefiltered[self] = []
        self.sent_data=0
        self.recv_data=0
        self.pickle_time = 0.0
        self.unpickle_time = 0.0
        #print ("Created Queue ADDR:",self.ip,self.port, "HINT:",hint_ip)

    def get_stats(self):
        return (self.ip, self.port, self.recv_data, self.sent_data, self.unpickle_time, self.pickle_time)


    def put(self,
            item: Any,
            block: bool = True,
            timeout: Optional[float] = None,
            raw: Optional[bool] =False) -> None:
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
        '''
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
        '''
        #t = -time.time()
        #msg = item if raw else pickle(item)
        #t += time.time()
        ##s.send(msg,copy=False)
        #s.send_multipart(msg,copy=False)
        ##s.send_multipart(msg)
        ##self.sent_data+=len(msg)
        #self.sent_data+=len(msg[0])
        #for d in msg[1:]:
        #    self.sent_data+=len(d.raw())
        #if not raw:
        #    self.pickle_time+=t
        #print("sending from", rank, "to", self.rank, "type",self.tag)
        comm.bsend(item, dest=self.rank, tag=self.tag)
        #print("send complete from", rank, "to", self.rank, "type",self.tag)


    def get(self,
            block: bool = True,
            gfilter = lambda x: True,
            timeout: Optional[float] = None,
            print_times: Optional[bool] = False,
            msginfo = lambda x: "") -> Any:
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
        for i in range(len(pf)):
            if gfilter(pf[i]):
                msg = pf[i]
                del pf[i]
                t1 = time.time()
                if print_times: print("Get msg:  from prefiltered ",(t1-t0)*1000, msginfo(msg))
                return msg
        while True:
            ##msg0 = s.recv()
            #msg0 = s.recv_multipart(copy=False)
            ##success=False
            ##while not success:
            ##    try:
            ##        msg0 = s.recv_multipart(copy=False,flags=zmq.NOBLOCK)
            ##        success=True
            ##    except:
            ##        sucess=False
            ##self.recv_data+=len(msg0)
            #for d in msg0:
            #    self.recv_data+=len(d)
            t1 = time.time()
            ##msg = cp.loads(msg0)
            ##msg = pick.loads(msg0)
            #msg = pick.loads(msg0[0],buffers=msg0[1:])
            ##msg = pyarrow.deserialize(msg0)
            t2 = time.time()
            #self.unpickle_time+=t2-t1
            msg = comm.recv(tag=self.tag)
            if gfilter(msg):
                #if print_times: print("Get msg: from queue ", len(msg0),"bytes",(t1-t0)*1000, "ms,  unpickle ", (t2-t1)*1000,"ms")
                if print_times: print("Get msg: from queue ", (t1-t0)*1000, "ms,  unpickle ", (t2-t1)*1000,"ms", msginfo(msg))
                return msg
            else:
                pf.append(msg)


    def multi_get(self,
            n: int = 1,
            gfilter = lambda x: True,
            timeout: Optional[float] = None,
            print_times=False,
            msginfo = lambda x: "") -> Any:
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
        if n<0:
            raise ValueError("'n' must be a non-negative number")
        return [self.get(gfilter=gfilter,print_times=print_times,msginfo=msginfo) for _ in range(n)]


