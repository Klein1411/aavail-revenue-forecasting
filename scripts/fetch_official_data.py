"""Download ONLY the public course-provided cs-train invoices from IBM case-study repo.

Requires Internet, and independently downloads directly from the upstream GitHub URL.
No third-party capstone solutions are copied.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from urllib.request import urlopen, Request
import io
import zipfile

URL = 'https://github.com/aavail/ai-workflow-capstone/archive/refs/heads/master.zip'
PREFIX = 'ai-workflow-capstone-master/cs-train/'


def download_train(target: str | Path = 'data/official') -> int:
    destination = Path(target)
    destination.mkdir(parents=True, exist_ok=True)
    request = Request(URL, headers={'User-Agent': 'AAVAIL-Capstone-Educational-Loader/1.0'})
    with urlopen(request, timeout=90) as response:
        raw = response.read(80 * 1024 * 1024 + 1)
    if len(raw) > 80 * 1024 * 1024:
        raise ValueError('Upstream zip exceeds 80 MiB limit')
    count = 0
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for info in archive.infolist():
            name = info.filename
            if not name.startswith(PREFIX) or not name.endswith('.json'):
                continue
            filename = Path(name).name
            if not filename.startswith('invoices-') or len(filename) > 80:
                continue
            if info.file_size > 20 * 1024 * 1024:
                raise ValueError('Invoice file exceeds size limit')
            (destination / filename).write_bytes(archive.read(info))
            count += 1
    if not count:
        raise ValueError('No training invoices found upstream; inspect repo layout')
    print(f'Downloaded {count} official AAVAIL invoice files to {destination}')
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/official')
    args = parser.parse_args()
    download_train(args.output)
