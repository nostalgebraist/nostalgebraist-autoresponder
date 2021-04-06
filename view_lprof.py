import sys
import line_profiler


def main(in_file="tumbl.py.lprof", out_file="lprof.txt"):
    lstats = line_profiler.load_stats(in_file)

    with open(out_file, 'w', encoding='utf-8') as f:
        line_profiler.show_text(lstats.timings, lstats.unit, output_unit=1e-2, stripzeros=True, stream=f)


if __name__ == "__main__":
    sys.exit(main())
