import json
import numpy as np

with open("data/nost_post_year_counts.json", "r") as f:
    nost_post_years_to_counts = json.load(f)

with open("data/nost_post_year_fracs.json", "r") as f:
    nost_post_years_to_fracs = json.load(f)

years = sorted(nost_post_years_to_counts.keys())
year_counts = [nost_post_years_to_counts[year] for year in years]
year_fracs = [nost_post_years_to_fracs[year] for year in years]


def sample_year() -> str:
    year = np.random.choice(years, p=year_fracs)

    # just in case
    year = str(year)

    return year


def substitute_year_v10(v10_timestamp: str, year: str) -> str:
    subbed = " ".join(v10_timestamp.split(" ")[:3]) + " " + year
    return subbed


def sample_and_substitute_year_v10(v10_timestamp: str) -> str:
    year = sample_year()
    subbed = substitute_year_v10(v10_timestamp=v10_timestamp, year=year)
    return subbed
