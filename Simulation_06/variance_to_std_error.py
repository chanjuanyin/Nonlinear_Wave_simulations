from __future__ import annotations

import csv
import math
from pathlib import Path


def convert_variance_to_std_error(
	input_path: Path, output_path: Path, sample_count: float
) -> None:
	scale = math.sqrt(sample_count)

	with input_path.open(newline="") as infile, output_path.open(
		"w", newline=""
	) as outfile:
		reader = csv.reader(infile)
		writer = csv.writer(outfile)

		for row in reader:
			if not row:
				writer.writerow([])
				continue

			converted = []
			for value in row:
				value = value.strip()
				if value == "":
					converted.append("")
					continue
				number = float(value)
				converted.append(f"{math.sqrt(number) / scale:.15g}")
			writer.writerow(converted)


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	input_path = base_dir / "results" / "monte_carlo_variance.csv"
	output_path = base_dir / "results" / "monte_carlo_std_error.csv"
	convert_variance_to_std_error(input_path, output_path, 1e7)


if __name__ == "__main__":
	main()
