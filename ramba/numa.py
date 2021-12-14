"""
Copyright 2021 Intel Corporation

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import psutil

try:
    import numa

    HAS_NUMA = numa.available()
except:
    HAS_NUMA = False


# Get number of numa zones -- Linux-specific!!


def get_zones(zones=None):
    if zones is None or zones == "":
        try:
            with open("/sys/devices/system/node/possible") as f:
                zones = f.read()
        except:
            return 0  # could not read
        numzones = int(zones[:-1].split("-")[-1]) + 1
    else:
        numzones = len(zones.split(":"))
    return numzones


# Get list of cpus in zone -- Linux-specific!!
def get_zone_cpus(z, zones=None):
    if zones is None or zones == "":
        with open("/sys/devices/system/node/node" + str(z) + "/cpulist") as f:
            cpuranges = f.read()[:-1].split(",")
    else:
        cpuranges = (zones.split(":")[z]).split(",")
    cpuranges = [x.split("-") for x in cpuranges]
    cpus = [i for x in cpuranges for i in range(int(x[0]), int(x[-1]) + 1)]
    return cpus


# set cpu affinity to a particular zone's cpus
# based on worker number;  assumes workers are allocated on one node before going to next -- Ray seems to do this so we should be good
def set_affinity_old(w, zones=None):
    nz = get_zones(zones)
    if nz == 0:
        return 0
    z = w % nz
    cpus = get_zone_cpus(z, zones)
    psutil.Process().cpu_affinity(cpus)
    # print ("Worker",w,"affinity",cpus)
    return z


# set cpu affinity to a particular zone's cpus, using numa.py;  fallback to old version if not available, or zones specified
# based on worker number;  assumes workers are allocated on one node before going to next -- Ray seems to do this so we should be good
def set_affinity(w, zones=None):
    if zones == "NONE" or zones == "DISABLE":
        return 0
    if not HAS_NUMA or (zones is not None and zones != ""):
        return set_affinity_old(w, zones)
    nz = numa.get_max_node() + 1
    z = w % nz
    numa.bind({z})
    # print ("Worker",w,"affinity",z)
    return z
