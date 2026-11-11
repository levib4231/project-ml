"""Export the implemented runtime for upload before it is available on GitHub."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ngps.provenance import source_record, write_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New ZIP path outside NGPS')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if Path(__file__).resolve().parents[1] in output.parents:
        parser.error('Export outside NGPS to avoid source fingerprint recursion')
    output.parent.mkdir(parents=True, exist_ok=True)
    record = source_record(output)
    write_json(output.with_suffix('.json'), record)
    print(output)
