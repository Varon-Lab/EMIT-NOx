import os
import earthaccess
import sys

import argparse

from config import CONFIG, LOCS
from REFERENCE_PLANTS import REFERENCE_PLANTS
from datetime import timedelta, datetime


def download_granules(results, data_folder, fs):
    for rf in results:
        result_urls = rf.data_links()
        for url in result_urls:
            granule_asset_id = url.split('/')[-1]
            
            # Download the Granule Asset if it doesn't exist
            fp = f'{data_folder}/{granule_asset_id}'
            if not os.path.isfile(fp):
                with fs.get(url,stream=True) as src:
                    with open(fp,'wb') as dst:
                        for chunk in src.iter_content(chunk_size=64*1024*1024):
                            dst.write(chunk)
            else:
                print(f"{url} already downloaded. Skipping...")
                continue
            print(f"DOWNLOADED {url} to {fp}")
    
def process_plants(args, fs):
    if not args.download:
        print("DOWNLOAD NOT REQUESTED.")
    
    for loc_name, md in REFERENCE_PLANTS.items():
        lon, lat = md['LON'], md['LAT']
        
        if args.L1B:
            L1B_results = earthaccess.search_data(
                short_name='EMITL1BRAD',
                point=(lon, lat),
                temporal=(args.starttime, args.endtime),
                cloud_cover=(0, args.cloud_cover),
                count=args.count
            )
            print(f"FOUND {len(L1B_results)} EMIT L1B GRANULES FOR {loc_name} AT {lat:0.3f}, {lon:0.3f}")
        else:
            L1B_results = []

        if args.L2A:
            L2A_results = earthaccess.search_data(
                    short_name='EMITL2ARFL',
                    point=(lon, lat),
                    temporal=(args.starttime, args.endtime),
                    cloud_cover=(0, args.cloud_cover),
                    count=args.count
                )
            print(f"FOUND {len(L2A_results)} EMIT L2A GRANULES FOR {loc_name} AT {lat:0.3f}, {lon:0.3f}")
        else:
            L2A_results = []
        
        results = L1B_results + L2A_results
        
        print(f"FOUND {len(results)} EMIT GRANULES FOR {loc_name} AT {lat:0.3f}, {lon:0.3f}")
        if args.download:
            save_path = f"{args.data_folder}/{loc_name}"
            os.makedirs(save_path, exist_ok=True)
            
            download_granules(results, save_path, fs)
            print("DONE DOWNLOADING GRANULES")

def process_list_of_plants(args, fs):
    if not args.download:
        print("DOWNLOAD NOT REQUESTED.")
    
    plantnames = [line.rstrip("\n") for line in open(args.plant_list)]

    for loc_name in plantnames:
        md = REFERENCE_PLANTS[loc_name]

        lon, lat = md['LON'], md['LAT']
        
        if args.L1B:
            L1B_results = earthaccess.search_data(
                short_name='EMITL1BRAD',
                point=(lon, lat),
                temporal=(args.starttime, args.endtime),
                cloud_cover=(0, args.cloud_cover),
                count=args.count
            )
            print(f"FOUND {len(L1B_results)} EMIT L1B GRANULES FOR {loc_name} AT {lat:0.3f}, {lon:0.3f}")
        else:
            L1B_results = []

        if args.L2A:
            L2A_results = earthaccess.search_data(
                    short_name='EMITL2ARFL',
                    point=(lon, lat),
                    temporal=(args.starttime, args.endtime),
                    cloud_cover=(0, args.cloud_cover),
                    count=args.count
                )
            print(f"FOUND {len(L2A_results)} EMIT L2A GRANULES FOR {loc_name} AT {lat:0.3f}, {lon:0.3f}")
        else:
            L2A_results = []
    
        results = L1B_results + L2A_results
        
        print(f"FOUND {len(results)} EMIT GRANULES FOR {loc_name} AT {lat:0.3f}, {lon:0.3f}")
        if args.download:
            save_path = f"{args.data_folder}/{loc_name}"
            os.makedirs(save_path, exist_ok=True)
            
            download_granules(results, save_path, fs)
            print("DONE DOWNLOADING GRANULES")

if __name__ == "__main__":
    auth = earthaccess.login(persist=True)
    fs = earthaccess.get_requests_https_session()
    
    if auth.authenticated:
        print("Authenticated to EarthData")
    else:
        raise Exception("Unable to authenticate to EarthData")
    
    parser = argparse.ArgumentParser(description="Download satellite data for a given location and time range.")

    # Required
    parser.add_argument("--lat", type=float, help="Latitude of the location")
    parser.add_argument("--lon", type=float, help="Longitude of the location")
    parser.add_argument("--loc_name", type=str, help="Location name")

    # Optional
    parser.add_argument(
        "--starttime",
        type=str,
        default="20220101",
        help="Start date in YYYYMMDD format (default: 20220101)"
    )
    parser.add_argument(
        "--endtime",
        type=str,
        default=datetime.now().strftime("%Y%m%d"),
        help="End date in YYYYMMDD format (default: current date)"
    )
    parser.add_argument(
        "--cloud_cover",
        type=int,
        default=100,
        help="Maximum cloud cover percentage (default: 90)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Maximum number of results to fetch (default: 100)"
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default=CONFIG['data_folder'],
        help='Path to data folder (default to config file)'
    )
    
    parser.add_argument("--download", action="store_true", help="Flag for data download")
    parser.add_argument("--grab_all_plants", action="store_true", help="Flag to grab all power plants")
    parser.add_argument("--L2A", action="store_true", help="Flag to pull L2A")
    parser.add_argument("--L1B", action="store_true", help="Flag to pull L1B")

    parser.add_argument(
        "--plant_list",
        type=str,
        default="",
        help='filepath to list of plants to download'
    )

    args = parser.parse_args()
    
    if not args.L2A and not args.L1B:
        raise Exception("Must specify either --L2A or --L1B")

    if args.grab_all_plants:
        process_plants(args, fs)
        quit()
    
    if args.plant_list:
        process_list_of_plants(args, fs)
        quit()
        
    if not ((args.lat and args.lon) or args.loc_name):
        raise Exception("Must specify either lat/lon or a location name")
        
    if args.loc_name in LOCS:
        lon = LOCS[args.loc_name]['LON']
        lat = LOCS[args.loc_name]['LAT']
        loc_name = args.loc_name
    elif args.loc_name in REFERENCE_PLANTS:
        lon = REFERENCE_PLANTS[args.loc_name]['LON']
        lat = REFERENCE_PLANTS[args.loc_name]['LAT']
        loc_name = args.loc_name
    else:
        lon = args.lon
        lat = args.lat
        loc_name = args.loc_name if args.loc_name else f"LOC_{lat:0.3f}_{lon:0.3f}"
    
    if args.L1B:
        L1B_results = earthaccess.search_data(
            short_name='EMITL1BRAD',
            point=(lon, lat),
            temporal=(args.starttime, args.endtime),
            cloud_cover=(0, args.cloud_cover),
            count=args.count
        )
        print(f"FOUND {len(L1B_results)} EMIT L1B GRANULES FOR {loc_name} AT {lat:0.3f}, {lon:0.3f}")
    else:
        L1B_results = []

    if args.L2A:
        L2A_results = earthaccess.search_data(
                short_name='EMITL2ARFL',
                point=(lon, lat),
                temporal=(args.starttime, args.endtime),
                cloud_cover=(0, args.cloud_cover),
                count=args.count
            )
        print(f"FOUND {len(L2A_results)} EMIT L2A GRANULES FOR {loc_name} AT {lat:0.3f}, {lon:0.3f}")
    else:
        L2A_results = []
    
    results = L1B_results + L2A_results
    print(f"FOUND {len(results)} EMIT GRANULES FOR {loc_name} AT {lat:0.3f}, {lon:0.3f}")
    if args.download:
        save_path = f"{args.data_folder}/{loc_name}"
        os.makedirs(save_path, exist_ok=True)
        
        download_granules(results, save_path, fs)
        print("DONE DOWNLOADING GRANULES")
    else:
        print("DOWNLOAD NOT REQUESTED. EXITING.")
        for l in L1B_results:
            print(l)
            print('\n\n\n')
