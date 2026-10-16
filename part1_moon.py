"""Part 1: plot moon illumination (2018) from moon_2018.csv and fit a sinusoid."""
from pathlib import Path
import pandas as pd
from sinefit import fit_sinusoid, print_summary, plot_fit

HERE = Path(__file__).parent
CSV_PATH = HERE / "moon_2018.csv"     # columns: day_of_year, date, fraction_illuminated

df = pd.read_csv(CSV_PATH)

# Project rule: all of January (days 1-31), then every day ending in 5 or 0 (35, 40, ... 365)
keep = (df["day_of_year"] <= 31) | ((df["day_of_year"] >= 35) & (df["day_of_year"] % 5 == 0))
df = df[keep].copy()
df["percent"] = df["fraction_illuminated"] * 100          # project wants percentages
print(f"Using {len(df)} data points")

x = df["day_of_year"].to_numpy()
y = df["percent"].to_numpy()

# The moon's cycle is ~29.5 days, so search a window around that
params = fit_sinusoid(x, y, period_range=(25, 35))
print_summary(params, xname="x", yname="y", units="%")

(HERE / "plots").mkdir(exist_ok=True)
plot_fit(x, y, params,
         title="Percentage of Moon Illuminated, 2018 (Jan 1-31, then every 5th day)",
         xlabel="Day of year (Jan 1 = day 1)",
         ylabel="Moon surface illuminated (%)",
         xmin=0, xmax=370,
         save_path=HERE / "plots" / "moon_fit.png")