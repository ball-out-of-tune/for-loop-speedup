"""Helper to compile CUDA with MSVC env."""
import subprocess, os, sys

msvc_dir = r'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.44.35207'
cl_dir = os.path.join(msvc_dir, 'bin', 'HostX64', 'x64')
include_dir = os.path.join(msvc_dir, 'include')
lib_dir = os.path.join(msvc_dir, 'lib', 'x64')

kit_dir = r'C:\Program Files (x86)\Windows Kits\10'
sdk_versions = os.listdir(os.path.join(kit_dir, 'Include'))
sdk_ver = [v for v in sdk_versions if v.startswith('10.')][0]
print(f'SDK: {sdk_ver}')

sdk_include = os.path.join(kit_dir, 'Include', sdk_ver)
sdk_lib = os.path.join(kit_dir, 'Lib', sdk_ver)

env = os.environ.copy()
env['PATH'] = cl_dir + ';' + env.get('PATH', '')
env['INCLUDE'] = ';'.join([
    include_dir,
    os.path.join(sdk_include, 'ucrt'),
    os.path.join(sdk_include, 'um'),
    os.path.join(sdk_include, 'shared'),
])
env['LIB'] = ';'.join([
    lib_dir,
    os.path.join(sdk_lib, 'ucrt', 'x64'),
    os.path.join(sdk_lib, 'um', 'x64'),
])

cwd = r'C:\Users\16874\codingPractice\numba-benchmark'
src = sys.argv[1] if len(sys.argv) > 1 else 'nosmem_faster.cu'
out = src.replace('.cu', '.exe')

cmd = ['nvcc', '-o', out, src, '-arch=sm_80']
print(f'Running: {" ".join(cmd)}')
result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
print('STDOUT:', result.stdout)
print('STDERR:', result.stderr)
print('RC:', result.returncode)
