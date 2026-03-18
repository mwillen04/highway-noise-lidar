import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
import osmnx as ox
import requests
import json
import os
import pdal
from tqdm import tqdm
from shapely import Polygon

def street2Point(roadDF: gpd.GeoDataFrame, interval_meters: float) -> gpd.GeoDataFrame:
    """
    This function generates random points for all streets with a 50-meter interval.

    Parameters:
    roadDF (GeoDataFrame): The GeoDataFrame containing the street network obtained in Part 2.

    Returns:
    GeoDataFrame: A DataFrame with three columns: [id, lon, lat], each row represents one point.
    """
    # Add a reference column for dissolving the street network into a single geometry
    roadDF['ref'] = 1

    # Dissolve the street network into a single geometry to calculate total length
    roadDDF = roadDF.dissolve(by='ref')

    # Define the interval distance in degrees (0.0005 = approximately 50 meters)
    interval = interval_meters / 111111

    # Calculate the total length of the dissolved geometry
    totalLength = roadDDF.geometry.length.iloc[0]

    # Initialize a list to store generated points
    pointList = []

    # Generate points at specified intervals along the total length
    for dist in np.arange(0, totalLength, interval):
        pointList.append(roadDDF.geometry.interpolate(dist))

    # Convert the list of points to a DataFrame
    pts = pd.DataFrame(pointList)
    pts['id'] = pts.index

    # Rename columns for clarity
    pts.columns = ['geometry', 'id']

    # Extract x and y coordinates from the geometry and round them
    pts["x"] = pts.geometry.apply(lambda x: str(x).split(' ')[1][1:])
    pts["y"] = pts.geometry.apply(lambda x: str(x).split(' ')[2][:-1])
    pts_all = pts.drop(["geometry", "id"], axis=1)

    pts_all.x = pts_all.x.astype('float64')
    pts_all.y = pts_all.y.astype('float64')
    pts_all.x = pts_all.x.apply(lambda x: np.round(x, 5))
    pts_all.y = pts_all.y.apply(lambda x: np.round(x, 5))

    # Convert the points to a GeoDataFrame and extract longitude and latitude
    gdf = gpd.points_from_xy(pts_all.x, pts_all.y)
    pts_shp = gpd.GeoDataFrame(pd.DataFrame(dict(geometry=gdf)))
    pts_shp["lng"] = pts_shp.geometry.x
    pts_shp["lat"] = pts_shp.geometry.y

    # Remove duplicate points and reset the index
    pts_shp = pts_shp.drop_duplicates(subset=["lng", "lat"])
    pts_shp = pts_shp.reset_index()
    pts_shp = pts_shp.drop(['geometry'], axis=1)
    pts_shp.columns = ["id", "lon", "lat"]

    return pts_shp

def create_target_area(place: str, buffer_dist: float = 1000) -> dict:

    # Get just the highways in `place`
    highway_network = ox.graph_from_place(place, network_type="drive", custom_filter='["highway"~"motorway"]')
    
    # Get just the edges
    highways = ox.graph_to_gdfs(highway_network)[1].to_crs(epsg=3857).reset_index()

    # Create a `buffer_dist` meter buffer around the highways
    highway_buffer = highways.dissolve().buffer(buffer_dist).to_crs(epsg=4326).iloc[0]
    highway_buffer = remove_invalid_holes(highway_buffer)

    bufferGDF = gpd.GeoDataFrame(index=[0], crs='EPSG:4326', geometry=[highway_buffer])
    
    # Get just the streets within the buffer
    target_area = ox.graph_from_polygon(highway_buffer, network_type="drive")

    # Convert the streets to a GeoDataFrame
    targetGDF = ox.graph_to_gdfs(target_area)[1].reset_index()
    targetGDF["is_highway"] = targetGDF["highway"].astype(str).str.contains("motorway")
    
    return {
        "highway_graph" : highway_network,
        "highwayGDF" : highways,
        "buffer_polygon" : highway_buffer,
        "bufferGDF" : bufferGDF,
        "target_graph" : target_area,
        "targetGDF" : targetGDF,
    }

def remove_invalid_holes(poly: Polygon, min_area: float = 1) -> Polygon:
    valid_holes = [ring.coords for ring in poly.interiors if Polygon(ring).area > min_area]
    out = Polygon(poly.exterior.coords, valid_holes)
    return out

def scores_from_json(pointsGDF: gpd.GeoDataFrame, hl_api_key: str, to_file: str) -> dict:

    if len(pointsGDF) > 250:
        print("Cannot process more than 250 points.")
        return None

    # Prep HowLoud query
    url = 'https://api.howloud.com/v2/score'
    headers = {'x-api-key': hl_api_key, 'Content-Type': 'application/json'}
    data = [
        {"id": str(i), "lat": lat, "lng": lon} for i, lat, lon in zip(pointsGDF["id"], pointsGDF["lat"], pointsGDF["lon"])
    ]
    print(data[:5])

    # # Query HowLoud
    r = requests.post(url, json=data, headers=headers)
    hl_scores = r.json()

    # Save a backup
    with open(to_file, 'w') as f:
        json.dump(hl_scores, f, indent=4)

    return hl_scores

def load_scores(from_file: str) -> gpd.GeoDataFrame:
    """
    This function retrieves HowLoud score points saved in a JSON file

    Parameters:
    from_file (str): Full pathname of the JSON file from which to pull the HowLoud points

    Returns:
    GeoDataFrame: A GeoDataFrame with HowLoud scores and categories; each row represents one point.
    """

    with open(from_file, 'r', encoding='utf-8') as f:
        scoresJSON = json.load(f)

    # Convert JSON to DataFrame and split out into columns
    scores = (pd.json_normalize(scoresJSON)
                .drop(columns=['result.request.lat', 'result.request.lng'])
                .explode("result.result"))

    # Nicer column names
    scores.columns = ["id", "lat", "lon", "details", "status"]
    details = scores['details'].apply(pd.Series)
    scoresDF = pd.concat([scores.drop('details', axis=1), details], axis=1)

    # Convert to GeoDataFrame
    scoresGDF = gpd.GeoDataFrame(scoresDF, geometry=gpd.points_from_xy(scoresDF.lon, scoresDF.lat, crs="EPSG:4326"))

    return scoresGDF

def network_distance(scoresGDF: gpd.GeoDataFrame, highwayGDF) -> gpd.GeoSeries:
    distances = scoresGDF.to_crs(3857).geometry.apply(lambda x: highwayGDF.distance(x).min())
    return distances

def map_data(base: gpd.GeoDataFrame, scores: gpd.GeoDataFrame, z: str, title: str, ax_args: dict = {"figsize":(10,7)}) -> None:
    fig, ax = plt.subplots(**ax_args)
    base.to_crs(3857).plot(ax=ax, column="is_highway", cmap="Set1_r", zorder=0)
    scores.to_crs(3857).plot(ax=ax, column=z, cmap="magma_r", markersize=50, 
                            legend=True, legend_kwds={"shrink": 0.75})
    
    plt.title(title)
    plt.show()

def get_lidar_tiles(tiles: list[tuple[str, str]], to_dir: str = "lidar_tiles") -> None:
    """
    Downloads all of the LiDAR tiles from the input list of tile files.

    :param tiles: A list of url-filename pairs; each corresponds to a LiDAR tile
    :type tiles: list
    :param to_dir: Output directory for the tile downloads
    :type to_dir: str
    """

    # Setup output folder, if needed
    save_directory = os.path.join("_data", to_dir)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for url, filename in tqdm(tiles):

        try:
            # Get file and set save path
            response = requests.get(url, stream=True)
            save_path = os.path.join(save_directory, filename)

            # Check for HTTPErrors
            response.raise_for_status()

            # Write output to file
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        except requests.exceptions.RequestException as e:
            print(f"An error occurred during the request: {e}")
        except IOError as e:
            print(f"An error occurred while writing the file: {e}")

