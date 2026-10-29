"""Refuse unsafe unified-memory startup, then exec the container's vLLM."""
import os
from pathlib import Path
import sys

GIB = 1024**3


def memory_budget(arguments, info):
    fraction = float(arguments[arguments.index('--gpu-memory-utilization')+1])
    if not 0 < fraction < 1:
        raise ValueError('Invalid GPU memory fraction')
    fields = {line.split(':')[0]: int(line.split()[1])*1024 for line in info.splitlines()
              if line.startswith(('MemTotal:', 'MemAvailable:'))}
    # Application import, CUDA context, loading/profile buffers and OS headroom.
    # This is a conservative admission check, not GPU memory isolation.
    return fields['MemAvailable'], int(fields['MemTotal']*fraction)+8*GIB


def main(arguments=None):
    arguments = sys.argv[1:] if arguments is None else arguments
    try:
        available, required = memory_budget(arguments, Path('/proc/meminfo').read_text())
    except (ValueError, KeyError, OSError, IndexError):
        print('Startup blocked: unified-memory admission check unavailable.', file=sys.stderr)
        return 78
    if available < required:
        print(f'Startup blocked: {available/GIB:.1f} GiB available; '
              f'{required/GIB:.1f} GiB required for the configured budget plus headroom. '
              'Existing services were left untouched.', file=sys.stderr)
        return 78
    os.execvp('vllm', ['vllm', 'serve', *arguments])


if __name__ == '__main__':
    raise SystemExit(main())
