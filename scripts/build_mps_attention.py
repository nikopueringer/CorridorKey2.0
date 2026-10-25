"""Build optional native attention with Xcode; no model conversion is required."""
from pathlib import Path
import platform
import subprocess
import sys
import torch

if sys.platform != 'darwin':
    raise SystemExit('Native MPS attention is only built on macOS')
if int(platform.mac_ver()[0].split('.')[0] or 0) < 15:
    raise SystemExit('Native MPS attention requires macOS 15 or later')
root = Path(__file__).resolve().parents[1]
package = Path(torch.__file__).resolve().parent
build = root / 'corridorkey/native/build'
build.mkdir(parents=True, exist_ok=True)
command = ['xcrun', 'clang++', '-std=c++17', '-O3', '-shared', '-fPIC', '-fobjc-arc',
           '-I'+str(package / 'include'), '-I'+str(package / 'include/torch/csrc/api/include'),
           str(root / 'corridorkey/native/mps_attention.mm'), '-L'+str(package / 'lib'),
           '-ltorch', '-ltorch_cpu', '-lc10', '-Wl,-rpath,'+str(package / 'lib'),
           '-framework', 'Foundation', '-framework', 'Metal',
           '-framework', 'MetalPerformanceShaders', '-framework', 'MetalPerformanceShadersGraph',
           '-o', str(build / 'mps_attention.so')]
subprocess.run(command, check=True)
(build / 'torch-version.txt').write_text(torch.__version__ + '\n')
print('Built:', build / 'mps_attention.so')
