import os, os.path as op
import sys
import zipfile
import subprocess
import multiprocessing
import pip
from shutil import copyfile


def check_packages(packagedict):
    installed_packages = [ x.project_name for x in pip.get_installed_distributions() ]
    for pkg_name in packagedict:
        if pkg_name not in installed_packages:
            pkg_dst = packagedict[pkg_name] if packagedict[pkg_name]!='' else pkg_name;
            pip.main(['install', pkg_dst])

#setup data
subprocess.call(["robocopy", "\\\\ivm-server2\\IRIS\\IRISObjectDetection\\Data\\datasets", 'data', '/e'], shell=True)
subprocess.call(["robocopy", '\\\\ivm-server2\\IRIS\\IRISObjectDetection\\Data\\imagenet_models', 'models', '*.caffemodel','/e'],shell=True)
packagedict = {'opencv-python':'' , 'progressbar':'', 'easydict':'', 'Cython':'', 'protobuf':''}
check_packages(packagedict);

msbuild = r'C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe'
assert os.path.exists(msbuild), 'Expect MSBuild.exe installed at %s' % msbuild

python_exe = sys.executable
print(python_exe)

if not 'PYTHON_ROOT' in os.environ:
    assert bool(python_exe), 'python.exe not found'
    os.environ['PYTHON_ROOT'] = os.path.split(sys.executable)[0]
    print('set PYTHON_ROOT to %s' % os.environ['PYTHON_ROOT'])

cwd = os.getcwd()
libpath = op.join(cwd, 'src','py-faster-rcnn','lib')
os.chdir(libpath)
gpunms_sln = op.join(libpath,'VCProjs','VCProjs.sln')
subprocess.call([python_exe, "setup.py", 'build_ext', '--inplace'], shell=True)
subprocess.call([msbuild, gpunms_sln, '/t:Build', '/p:Configuration=Release'], shell=True)

os.chdir(cwd)
