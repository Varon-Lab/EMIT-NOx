#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import json

def main():
    parser = argparse.ArgumentParser(description="Top N NOx emitters with coordinates")
    parser.add_argument("N", type=int, help="Number of facilities to output")
    args = parser.parse_args()
    N = args.N

    # Load CSVs
    emissions = pd.read_csv("annual-emissions-facility.csv")
    facilities = pd.read_csv("facility-attributes.csv")
    facilities = facilities.drop_duplicates('Facility ID')

    # Convert NOx column to numeric and drop invalid rows
    emissions["NOx Mass (short tons)"] = pd.to_numeric(
        emissions["NOx Mass (short tons)"], errors="coerce"
    )
    emissions = emissions.dropna(subset=["NOx Mass (short tons)"])

    # Merge on Facility ID
    merged = emissions.merge(facilities, on="Facility ID", how="inner")
    
    print(merged.columns)
    print(np.unique(merged['Facility ID'], return_counts=True))
    # Sort by NOx descending and select top N
    top = merged.sort_values("NOx Mass (short tons)", ascending=False).head(N)
    
    print(top)
    # Build dictionary with underscores in facility names
#    print(row.keys())s
    result = {
        row["Facility Name_x"].replace(" ", "_"): {"LON": row["Longitude"],
        "LAT": row["Latitude"],
        "NOx Mass (short tons)": row["NOx Mass (short tons)"],
        "State": row["State_x"],
        "Fac_ID": row["Facility ID"]}
        for _, row in top.iterrows()
    }

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

