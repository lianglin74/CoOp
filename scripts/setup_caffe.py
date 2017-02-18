import os
import sys
import zipfile
import subprocess
import multiprocessing

msbuild = r'C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe'
assert os.path.exists(msbuild), 'Expect MSBuild.exe installed at %s' % msbuild
cpu_count = multiprocessing.cpu_count()

if not 'PYTHON_ROOT' in os.environ:
    python_exe = sys.executable
    assert bool(python_exe), 'python.exe not found'
    os.environ['PYTHON_ROOT'] = os.path.split(sys.executable)[0]
    print('set PYTHON_ROOT to %s' % os.environ['PYTHON_ROOT'])

cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, 'src/CCSCaffe/scripts'))

import download_prebuilt_dependencies
import install_cudnn

os.chdir('src/CCSCaffe')

if not os.path.exists('libraries'):
    download_prebuilt_dependencies.main([])
else:
    print('Skip install prebuilt dependencies. For clean setup, please delete folder <src/CCSCaffe/libraries> and run this script again')

if not os.path.exists('cudnn'):
    #cudnn_zipfile = r'\\ivm-server2\IRIS\IRISObjectDetection\Data\cudnn-8.0-windows10-x64-v5.0-ga.zip'
    cudnn_zipfile = r'\\ivm-server2\IRIS\IRISObjectDetection\Data\cudnn-8.0-windows10-x64-v5.1.zip'
    install_cudnn.main([cudnn_zipfile])
else:
    print('Skip install cudnn. For clean setup, please delete folder <src/CCSCaffe/cudnn> and run this script again')

print('\n')
subprocess.call([msbuild, 'caffe.sln', '/t:Build', '/p:Configuration=Release', '/maxcpucount:%d' % max(1, cpu_count - 4)], shell=True)

os.chdir(cwd)
