# ramba
Python combination of Ray and Numba providing compiled distributed arrays, remote functions, and actors.

Please note that this work is a research prototype and that it internally uses Ray and/or ZeroMQ for
communication.  These communication channels are generally not secured or authenticated.  This means
that data sent across those communication channels may be visible to eavesdroppers.  Also, it is means
that malicious users may be able to send messages to this system that are interpreted as legitimate.
This may result in corrupted data and since pickled functions are also sent over the communication
channel, malicious attackers may be able to run arbitrary code.

Since this prototype uses Ray, occasionally orphaned Ray processes may be left around.  These can
be stopped with the command "ray stop".
