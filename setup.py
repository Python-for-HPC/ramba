"""
Copyright 2021 Intel Corporation

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import sys
from distutils.command import build
from distutils.command.build_ext import build_ext
from distutils.spawn import spawn
from setuptools import find_packages, setup
import versioneer

_version_module = None
try:
    from packaging import version as _version_module
except ImportError:
    try:
        from setuptools._vendor.packaging import version as _version_module
    except ImportError:
        pass


min_python_version = "3.7"
max_python_version = "3.10"  # exclusive
min_numpy_build_version = "1.11"
min_numpy_run_version = "1.15"


def _guard_py_ver():
    if _version_module is None:
        return

    parse = _version_module.parse

    min_py = parse(min_python_version)
    max_py = parse(max_python_version)
    cur_py = parse(".".join(map(str, sys.version_info[:3])))

    if not min_py <= cur_py < max_py:
        msg = (
            "Cannot install on Python version {}; only versions >={},<{} "
            "are supported."
        )
        raise RuntimeError(msg.format(cur_py, min_py, max_py))


_guard_py_ver()


class build_doc(build.build):
    description = "build documentation"

    def run(self):
        spawn(["make", "-C", "docs", "html"])


versioneer.VCS = "git"
versioneer.versionfile_source = "ramba/_version.py"
versioneer.versionfile_build = "ramba/_version.py"
versioneer.tag_prefix = ""
versioneer.parentdir_prefix = "ramba-"

cmdclass = versioneer.get_cmdclass()
cmdclass["build_doc"] = build_doc

install_name_tool_fixer = []

build_ext = cmdclass.get("build_ext", build_ext)

ramba_be_user_options = []


class RambaBuildExt(build_ext):

    user_options = build_ext.user_options + ramba_be_user_options
    boolean_options = build_ext.boolean_options + ["werror", "wall", "noopt"]

    def initialize_options(self):
        super().initialize_options()
        self.werror = 0
        self.wall = 0
        self.noopt = 0

    def run(self):
        extra_compile_args = []
        if self.noopt:
            if sys.platform == "win32":
                extra_compile_args.append("/Od")
            else:
                extra_compile_args.append("-O0")
        if self.werror:
            extra_compile_args.append("-Werror")
        if self.wall:
            extra_compile_args.append("-Wall")
        for ext in self.extensions:
            ext.extra_compile_args.extend(extra_compile_args)

        super().run()


cmdclass["build_ext"] = RambaBuildExt


def is_building():
    """
    Parse the setup.py command and return whether a build is requested.
    If False is returned, only an informational command is run.
    If True is returned, information about C extensions will have to
    be passed to the setup() function.
    """
    if len(sys.argv) < 2:
        # User forgot to give an argument probably, let setuptools handle that.
        return True

    build_commands = [
        "build",
        "build_py",
        "build_ext",
        "build_clib" "build_scripts",
        "install",
        "install_lib",
        "install_headers",
        "install_scripts",
        "install_data",
        "sdist",
        "bdist",
        "bdist_dumb",
        "bdist_rpm",
        "bdist_wininst",
        "check",
        "build_doc",
        "bdist_wheel",
        "bdist_egg",
        "develop",
        "easy_install",
        "test",
    ]
    return any(bc in sys.argv[1:] for bc in build_commands)


def get_ext_modules():
    """
    Return a list of Extension instances for the setup() call.
    """
    # Note we don't import Numpy at the toplevel, since setup.py
    # should be able to run without Numpy for pip to discover the
    # build dependencies
    ext_modules = []

    return ext_modules


packages = find_packages()

build_requires = ["numpy >={}".format(min_numpy_build_version)]
install_requires = [
    "numpy >={}".format(min_numpy_run_version),
    "setuptools",
]

metadata = dict(
    name="ramba",
    description="combining Ray + Numba",
    version=versioneer.get_version(),
    classifiers=[],
    package_data={},
    scripts=[],
    author="Intel, Inc.",
    author_email="todd.a.anderson@intel.com",
    packages=packages,
    setup_requires=build_requires,
    install_requires=install_requires,
    python_requires=">={},<{}".format(min_python_version, max_python_version),
    license="BSD",
    cmdclass=cmdclass,
)

if is_building():
    metadata["ext_modules"] = get_ext_modules()

setup(**metadata)
