#!/usr/bin/env python

# from shutil import copyfile, copymode
# import sysconfig
import platform
# 
from distutils.version import LooseVersion
# from setuptools import setup, find_packages, Extension
# from setuptools.command.build_ext import build_ext

import subprocess
import os
import re
import sys
from setuptools import setup

from distutils.cmd import Command

class CMakeBuild(Command):
    description = "Description of the command"
    user_options = []

    # This method must be implemented
    def initialize_options(self):
        pass

    # This method must be implemented
    def finalize_options(self):
        pass

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                                   out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")
        self.build()

    def build(self):
        self.cwd = os.path.abspath(os.path.dirname(__file__))
        print('Working directory:' + ' ' + self.cwd)
        self.source_abspath = self.cwd
        self.build_abspath  = os.path.join(self.cwd, 'build')
        cmake_args = []

        cfg = 'Debug' # or 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            #cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
            #    cfg.upper(),
            #    extdir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']
        # env.get() -- can modify env variables
        env = os.environ.copy()

        # create build dir
        if not os.path.exists(self.build_abspath):
            os.makedirs(self.build_abspath)

        # generate project
        subprocess.check_call(['cmake', self.source_abspath] + cmake_args,
                              cwd=self.build_abspath, env=env)
        # build project
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_abspath)
        # Copy *_test file to tests directory
        #test_bin = os.path.join(self.build_temp, 'Release\\python_cpp_example_test.exe')
        #self.copy_test_file(test_bin)
        print()  # Add an empty line for cleaner output

    def copy_test_file(self, src_file):
        # Create directory if needed
        dest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests', 'bin')
        if dest_dir != "" and not os.path.exists(dest_dir):
            print("creating directory {}".format(dest_dir))
            os.makedirs(dest_dir)

        # Copy file
        dest_file = os.path.join(dest_dir, os.path.basename(src_file))
        print("copying {} -> {}".format(src_file, dest_file))
        #copyfile(src_file, dest_file)
        #copymode(src_file, dest_file)

mycmdclass = {}
mycmdclass['build_cmake'] = CMakeBuild

setup(
    name='driver-drowsiness',
    version='0.1',
    author='Drowsy Team',
    author_email='Drowsy_Team@gmail.com',
    description='Drowsiness alert project',
    long_description='',
    # tell setuptools to look for any packages under 'src'
    #packages=find_packages('src'),
    # tell setuptools that all packages will be under the 'src' directory
    # and nowhere else
    package_dir={'': 'src'},
    packages=['driver-drowsiness.Detector'],
    # add custom commands
    cmdclass=mycmdclass,
    zip_safe=False,
    test_suite='tests.driver-drowsiness'
)
