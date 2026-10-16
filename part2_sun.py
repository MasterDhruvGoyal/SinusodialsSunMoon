"""
Part 2: download sunrise/sunset for any city, compute daylight hours, fit a sinusoid.

Change CITY below (and COUNTRY_CODE if the name is ambiguous), or run from a terminal:
    python part2_sun.py "Singapore"
    python part2_sun.py "Houston" US
Data source: Open-Meteo (free, no API key). Geocoding turns the city name into latitude/longitude,
then the sunrise/sunset URL is built from those.
"""
import re
import sys
from pathlib import Path

import pandas as pd
import requests

from sinefit import fit_sinusoid, print_summary, plot_fit

# ======================= CHANGE ONLY THESE =======================
CITY = "Singapore"  # city name (case-insensitive); can include spaces, e.g. "New York"
COUNTRY_CODE = ""        # optional 2-letter code, e.g. "US" for Houston, "SG" for Singapore
YEAR = 2020
FIXED_PERIOD = 365       # a year; set to None to let the fit choose the period itself
# =================================================================

if len(sys.argv) > 1:
    CITY = sys.argv[1]
if len(sys.argv) > 2:
    COUNTRY_CODE = sys.argv[2]

HERE = Path(__file__).parent
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
SUN_URL = "https://archive-api.open-meteo.com/v1/archive"


def geocode(city, country_code=""):
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    if country_code:
        params["countryCode"] = country_code
    r = requests.get(GEOCODE_URL, params=params, timeout=30)
    r.raise_for_status()
    results = r.json().get("results")
    if not results:
        sys.exit(f"Could not find a city called '{city}'. Check the spelling or add COUNTRY_CODE.")
    return results[0]


def fetch_sun_table(lat, lon, year):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
        "daily": "sunrise,sunset", "timezone": "auto",
    }
    r = requests.get(SUN_URL, params=params, timeout=60)
    r.raise_for_status()
    print("Data URL:", r.url)
    d = r.json()["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(pd.Series(d["time"])),
        "sunrise": pd.to_datetime(pd.Series(d["sunrise"])),
        "sunset": pd.to_datetime(pd.Series(d["sunset"])),
    }).dropna()
    df["day_of_year"] = df["date"].dt.dayofyear
    # daylight = sunset - sunrise, as a decimal number of hours (computed from whole minutes)
    minutes = (df["sunset"] - df["sunrise"]).dt.total_seconds().round() / 60
    df["daylight_hours"] = minutes / 60
    return df.reset_index(drop=True)


def describe_day(label, row, ties):
    sr, ss = row["sunrise"], row["sunset"]
    total_min = int(round((ss - sr).total_seconds() / 60))
    h, m = divmod(total_min, 60)
    print(f"\n{label}: {row['date']:%B %d, %Y} (day {row['day_of_year']})")
    print(f"  Sunrise {sr:%H:%M}   Sunset {ss:%H:%M}")
    print(f"  Calculation: {ss:%H:%M} - {sr:%H:%M} = {h} h {m} min = {h} + {m}/60 = {total_min / 60:.4f} hours")
    if ties > 1:
        print(f"  (note: {ties} days tie for this value to the minute (first one shown))")


def main():
    place = geocode(CITY, COUNTRY_CODE)
    name = ", ".join(str(p) for p in (place.get("name"), place.get("admin1"), place.get("country")) if p)
    lat, lon = place["latitude"], place["longitude"]
    print(f"City: {name}  (lat {lat:.4f}, lon {lon:.4f}, timezone {place.get('timezone', '?')})")

    df = fetch_sun_table(lat, lon, YEAR)
    slug = re.sub(r"[^a-z0-9]+", "_", CITY.lower()).strip("_")
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "plots").mkdir(exist_ok=True)
    out = df.assign(sunrise=df["sunrise"].dt.strftime("%H:%M"), sunset=df["sunset"].dt.strftime("%H:%M"),
                    date=df["date"].dt.strftime("%Y-%m-%d"))
    csv_path = HERE / "data" / f"sun_{slug}_{YEAR}.csv"
    out[["date", "day_of_year", "sunrise", "sunset", "daylight_hours"]].to_csv(csv_path, index=False)
    print(f"Saved {len(df)} days of data -> {csv_path}")

    mx, mn = df["daylight_hours"].max(), df["daylight_hours"].min()
    describe_day("LONGEST day", df.loc[df["daylight_hours"].idxmax()], int((df["daylight_hours"] == mx).sum()))
    describe_day("SHORTEST day", df.loc[df["daylight_hours"].idxmin()], int((df["daylight_hours"] == mn).sum()))
    print(f"\nMaximum daylight: {mx:.4f} hours   Minimum daylight: {mn:.4f} hours")
    print(f"Amplitude check = (max - min)/2 = {(mx - mn) / 2:.4f};  midline check = (max + min)/2 = {(mx + mn) / 2:.4f}\n")

    x = df["day_of_year"].to_numpy()
    y = df["daylight_hours"].to_numpy()
    params = fit_sinusoid(x, y, period_range=(330, 400), fixed_period=FIXED_PERIOD)
    print_summary(params, xname="x", yname="y", units="hours")

    plot_fit(x, y, params,
             title=f"Hours of Daylight in {name}, {YEAR}",
             xlabel=f"Day of year ({YEAR}; Jan 1 = day 1)",
             ylabel="Hours of daylight",
             xmin=0, xmax=int(x.max()) + 1,
             save_path=HERE / "plots" / f"daylight_{slug}_{YEAR}.png",
             point_size=10)


if __name__ == "__main__":
    main()