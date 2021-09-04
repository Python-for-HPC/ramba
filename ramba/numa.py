"""
Copyright 2021 Intel Corporation

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import psutil

# Get number of numa zones -- Linux-specific!!

def get_zones():
    with open("/sys/devices/system/node/possible") as f:
        zones = f.read()
    numzones = int(zones[:-1].split('-')[-1])+1
    return numzones

# Get list of cpus in zone -- Linux-specific!!
def get_zone_cpus(z):
    with open("/sys/devices/system/node/node"+str(z)+"/cpulist") as f:
        cpuranges = f.read()[:-1].split(',')
    cpuranges = [ x.split('-') for x in cpuranges ]
    cpus = [ i for x in cpuranges for i in range(int(x[0]),int(x[-1])+1) ]
    return cpus

# set cpu affinity to a particular zone's cpus
# based on worker number;  assumes workers are allocated on one node before going to next -- Ray seems to do this so we should be good
def set_affinity(w):
    nz = get_zones()
    z = w % nz
    cpus = get_zone_cpus(z)
    psutil.Process().cpu_affinity(cpus)
    return z


