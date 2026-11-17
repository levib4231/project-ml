"""Verify all local MINC sources against freshly downloaded BrainWeb volumes.

Keep the HTTP receipts and hashes. They do not establish a redistribution
license. The script uses MINC1 headers to audit the historical NiBabel
conversion without rewriting it.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

import numpy as np
from scipy.io import netcdf_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.manifest import load_manifest, manifest_hash, resolve_path
from datasets.preparation import file_hash
from ngps.provenance import write_json
from ngps.checkpoint2 import SPLITS

BASE = 'https://brainweb.bic.mni.mcgill.ca'
LIMITATION = ('Public T1 images are fixed reference signals, not demonstrated noise-free anatomy. '
              'MINC history refers to pn3_rf20 coil settings. Metrics quantify recovery of these '
              'references after added Gaussian noise, not clinical effectiveness or removal of native artifacts.')


def acquire(source_manifest, cache):
    """Reacquire all public sources into a NEW directory on a hosted runtime."""
    path = Path(source_manifest).resolve()
    path.parent.mkdir(parents=True, exist_ok=False)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    records = []
    for split, subjects in SPLITS.items():
        for subject in subjects:
            data = urllib.parse.urlencode({'do_download_alias': subject + '_t1w', 'format_value': 'minc',
                                          'zip_value': 'gnuzip', 'download_for_real': '[Start download!]'}).encode()
            request = urllib.request.Request(BASE + '/cgi/brainweb1', data=data)
            with urllib.request.urlopen(request, timeout=180) as response:
                body = response.read()
            raw_bytes = gzip.decompress(body)
            (cache / (subject + '.mnc.gz')).write_bytes(body)
            (path.parent / (subject + '_t1w_p4.mnc.gz')).write_bytes(body)
            with netcdf_file(io.BytesIO(raw_bytes), mmap=False) as nc:
                image = nc.variables['image']
                if image.dimensions != ('zspace', 'yspace', 'xspace'):
                    raise ValueError('Unexpected MINC orientation')
                low, high = np.asarray(image.valid_range, dtype=float)
                lo = nc.variables['image-min'].data.copy()[:, None, None]
                hi = nc.variables['image-max'].data.copy()[:, None, None]
                physical = ((image.data.astype(np.float64) - low) / (high - low) * (hi - lo) + lo).astype(np.float32)
            volume = (np.clip(physical, 0, 4095) / np.float32(4095)).astype(np.float32)
            array = path.parent / (subject + '_clean.npy')
            np.save(array, volume)
            records.append({'subject_id': subject, 'split': split, 'source': array.name,
                            'source_sha256': file_hash(array), 'source_dtype': str(volume.dtype),
                            'source_shape': list(volume.shape),
                            'source_range': [float(volume.min()), float(volume.max())]})
    write_json(path, {'schema_version': 1, 'kind': 'source', 'seed': 42,
                     'description': 'Authoritative BrainWeb T1 fixed reference signals; MINC real /4095 course scale',
                     'subjects': sorted(records, key=lambda row: row['subject_id'])})
    return path


def verify(source_manifest, output, cache, accept_fixed_reference=False):
    if not accept_fixed_reference:
        raise ValueError('Explicit --accept-fixed-reference is required; see reference limitation')
    manifest = load_manifest(source_manifest)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    if Path(output).exists():
        raise FileExistsError(output)

    def one(row):
        subject = row['subject_id']
        alias = subject + '_t1w'
        page = BASE + '/cgi/brainweb1?' + urllib.parse.urlencode({'alias': alias, 'download': 1})
        form = urllib.request.urlopen(page, timeout=120).read().decode()
        if alias not in form or '0 to 4095' not in form:
            raise ValueError('Unexpected BrainWeb source page: ' + subject)
        (cache / (subject + '.html')).write_text(form)
        remote_path = cache / (subject + '.mnc.gz')
        if not remote_path.exists():
            data = urllib.parse.urlencode({'do_download_alias': alias, 'format_value': 'minc',
                                          'zip_value': 'gnuzip', 'download_for_real': '[Start download!]'}).encode()
            request = urllib.request.Request(BASE + '/cgi/brainweb1', data=data)
            with urllib.request.urlopen(request, timeout=180) as response:
                body = response.read()
            # Reject HTML/error responses before persisting any download.
            gzip.decompress(body)
            remote_path.write_bytes(body)
        local_array = resolve_path(source_manifest, row['source'])
        local_minc = local_array.with_name(subject + '_t1w_p4.mnc.gz')
        remote_bytes = gzip.decompress(remote_path.read_bytes())
        local_bytes = gzip.decompress(local_minc.read_bytes()) if local_minc.exists() else None
        # A mismatch is a verification failure, never overwrite the historical source.
        if local_bytes != remote_bytes:
            raise ValueError('Authoritative MINC differs from local source: ' + subject)
        with netcdf_file(io.BytesIO(remote_bytes), mmap=False) as nc:
            image = nc.variables['image']
            if image.dimensions != ('zspace', 'yspace', 'xspace'):
                raise ValueError('Unexpected MINC orientation')
            low, high = np.asarray(image.valid_range, dtype=float)
            raw = image.data.astype(np.float64)
            lo = nc.variables['image-min'].data.copy()[:, None, None]
            hi = nc.variables['image-max'].data.copy()[:, None, None]
            physical = ((raw - low) / (high - low) * (hi - lo) + lo).astype(np.float32)
            expected = (np.clip(physical, 0, 4095) / np.float32(4095)).astype(np.float32)
            actual = np.load(local_array, allow_pickle=False)
            if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-6, atol=1e-8):
                raise ValueError('Historical conversion mismatch: ' + subject)
            orientation = {axis: {'step_mm': float(nc.variables[axis].step),
                                  'start_mm': float(nc.variables[axis].start)}
                           for axis in image.dimensions}
            history = nc.history.decode()
        print('Verified authoritative source and conversion:', subject, flush=True)
        return {'subject_id': subject, 'source_sha256': file_hash(local_array),
                'source_page': page, 'download_post_url': BASE + '/cgi/brainweb1',
                'alias': alias, 'format': 'minc', 'compression': 'gnuzip',
                'authoritative_gzip_sha256': file_hash(remote_path),
                'decompressed_minc_sha256': hashlib.sha256(remote_bytes).hexdigest(),
                'local_minc_sha256': file_hash(local_minc), 'orientation': orientation,
                'history': history, 'conversion': 'MINC real scaling; float32; clip [0,4095]; divide by 4095',
                'conversion_max_abs_error': float(np.max(np.abs(actual - expected)))}

    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(one, manifest['subjects']))
    result = {'schema_version': 1, 'status': 'verified_fixed_reference',
              'verified_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_manifest_hash': manifest_hash(manifest),
              'normalization': {'policy': 'already-unit', 'historical_real_intensity_divisor': 4095,
                                'note': 'Course fixed scale, not a claim that MINC real intensity spans 0..4095.'},
              'dataset_page': BASE + '/brainweb/anatomic_normal_20.html',
              'terms_page': BASE + '/brainweb/#references',
              'terms': 'Public download with requested citations; no explicit redistribution license established. Obtain data directly; do not bundle volumes.',
              'reference_limitation': LIMITATION, 'subjects': records}
    write_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--accept-fixed-reference', action='store_true')
    parser.add_argument('--acquire', action='store_true', help='Create a NEW source directory/manifest from public downloads first')
    args = parser.parse_args()
    if args.acquire:
        if not args.accept_fixed_reference:
            parser.error('--acquire requires --accept-fixed-reference')
        acquire(args.manifest, args.cache)
    verify(args.manifest, args.output, args.cache, args.accept_fixed_reference)
