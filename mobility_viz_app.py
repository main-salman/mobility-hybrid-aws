import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
from datetime import datetime, timedelta
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import os
from pathlib import Path
import logging
from folium.plugins import HeatMap, MarkerCluster, FastMarkerCluster, TimestampedGeoJson, HeatMapWithTime, AntPath
import json
from typing import Dict, List, Tuple, Optional, Any
import time
import re
from scipy.spatial import KDTree
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, Rbf, UnivariateSpline
import matplotlib.cm as cm
from matplotlib.colors import to_hex
from itertools import groupby
from operator import itemgetter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CENTER = [37.7749, -122.4194]  # San Francisco by default
DEFAULT_ZOOM = 12
MAX_POINTS_PER_VIEW = 1000  # Reduced from 5000 to improve performance
CACHE_TTL = 3600  # 1 hour in seconds
MAX_MESSAGE_SIZE = 20 * 1024 * 1024  # 20MB in bytes
TILE_SIZE = 256  # Standard tile size
MAX_ZOOM = 18
MIN_ZOOM = 3
TIME_WINDOW_HOURS = 1  # Time window for animation chunks
SAMPLE_RATE = 0.01  # Sample 1% of data from each file
MAX_FILES_PER_DATE = 5  # Max number of files to process per date folder

# Animation settings
ANIMATION_INTERVAL = "2h"  # 2 hour intervals for smoother transitions (using 'h' instead of 'H')
MAX_ANIMATION_POINTS = 100  # Reduced for better performance
ANIMATION_SPEED = 1000  # Milliseconds between frames
TRANSITION_TIME = 800  # Transition time in ms (increased for smoother effect)
INTERPOLATE_GAPS = True  # Fill gaps in animation data
INTERPOLATION_SMOOTHNESS = 50  # Number of points to interpolate between real data points
USE_TRAILS = True  # Show movement trails for better visualization
MAX_PATH_LENGTH = 5  # Maximum number of previous points to show in path
POINT_PERSISTENCE = 3  # Number of frames a point remains visible

# Set page config
st.set_page_config(
    page_title="City Mobility Data Visualization",
    page_icon="🌆",
    layout="wide"
)

def get_date_from_folder(folder_name: str) -> datetime:
    """
    Extract date from folder name
    """
    date_str = folder_name.split('=')[1]
    return datetime.strptime(date_str, '%Y-%m-%d')

@st.cache_data(ttl=CACHE_TTL)
def load_data_for_range(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Load data for specific date range
    """
    try:
        logger.info(f"Loading data for date range: {start_date.date()} to {end_date.date()}")
        
        # Get list of date folders
        all_folders = [f for f in os.listdir('data') if f.startswith('date=')]
        
        # Filter folders by date range
        date_folders = []
        for folder in all_folders:
            folder_date = get_date_from_folder(folder)
            if start_date.date() <= folder_date.date() <= end_date.date():
                date_folders.append(folder)
        
        if not date_folders:
            logger.warning(f"No date folders found in range: {start_date.date()} to {end_date.date()}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(date_folders)} date folders in range")
        
        # Process each date folder
        dfs = []
        for date_folder in date_folders:
            logger.info(f"Processing folder: {date_folder}")
            
            # Get the date from folder name
            folder_date = date_folder.split('=')[1]
            
            # Get list of files in the folder
            folder_path = os.path.join('data', date_folder)
            parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
            
            # Sample files if there are too many
            if len(parquet_files) > MAX_FILES_PER_DATE:
                parquet_files = np.random.choice(parquet_files, MAX_FILES_PER_DATE, replace=False)
            
            logger.info(f"Processing {len(parquet_files)} files from {date_folder}")
            
            # Process each file
            for parquet_file in parquet_files:
                file_path = os.path.join(folder_path, parquet_file)
                
                # Read the file with sampling
                try:
                    table = pq.read_table(
                        file_path,
                        columns=['utc_timestamp', 'latitude', 'longitude']
                    )
                    
                    # Convert to pandas and sample
                    file_df = table.to_pandas()
                    sampled_df = file_df.sample(frac=SAMPLE_RATE)
                    
                    # Add date from folder name
                    sampled_df['folder_date'] = folder_date
                    
                    dfs.append(sampled_df)
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue
        
        # Combine all dataframes
        if not dfs:
            logger.error("No data loaded from any folder")
            return pd.DataFrame()
        
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(df)} sampled records")
        
        # Convert columns
        df['timestamp'] = pd.to_datetime(df['utc_timestamp'])
        
        # Convert lat/long to float
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # Drop rows with invalid lat/long
        df = df.dropna(subset=['latitude', 'longitude'])
        logger.info(f"After cleaning: {len(df)} valid records")
        
        # Add count column for visualization
        df['count'] = 1
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL)
def get_available_date_range() -> Tuple[datetime, datetime]:
    """
    Get available date range from folder names
    """
    try:
        # Get list of date folders
        date_folders = [f for f in os.listdir('data') if f.startswith('date=')]
        
        if not date_folders:
            logger.error("No date folders found")
            return datetime.now(), datetime.now()
        
        # Extract dates from folder names
        dates = []
        for folder in date_folders:
            try:
                date_str = folder.split('=')[1]
                date = datetime.strptime(date_str, '%Y-%m-%d')
                dates.append(date)
            except Exception as e:
                logger.error(f"Error parsing date from folder {folder}: {e}")
                continue
        
        if not dates:
            logger.error("No valid dates found")
            return datetime.now(), datetime.now()
        
        min_date = min(dates)
        max_date = max(dates)
        logger.info(f"Available date range: {min_date.date()} to {max_date.date()}")
        
        return min_date, max_date
    except Exception as e:
        logger.error(f"Error getting date range: {e}")
        return datetime.now(), datetime.now()

def aggregate_data(df: pd.DataFrame, zoom_level: int) -> pd.DataFrame:
    """
    Aggregate data based on zoom level to reduce resolution
    """
    if df.empty:
        return df
    
    # Calculate grid size based on zoom level
    grid_size = 2 ** (MAX_ZOOM - zoom_level)
    
    # Round coordinates to grid
    df['grid_lat'] = (df['latitude'] * grid_size).round() / grid_size
    df['grid_lon'] = (df['longitude'] * grid_size).round() / grid_size
    
    # Aggregate by grid cell
    aggregated = df.groupby(['grid_lat', 'grid_lon']).agg({
        'count': 'sum'
    }).reset_index()
    
    # Rename columns back
    aggregated['latitude'] = aggregated['grid_lat']
    aggregated['longitude'] = aggregated['grid_lon']
    aggregated = aggregated.drop(['grid_lat', 'grid_lon'], axis=1)
    
    return aggregated

def sample_data(df: pd.DataFrame, max_points: int = MAX_POINTS_PER_VIEW) -> pd.DataFrame:
    """
    Sample data if it exceeds the maximum number of points
    """
    if len(df) <= max_points:
        return df
    
    logger.info(f"Sampling data from {len(df)} to {max_points} points")
    return df.sample(n=max_points, random_state=42)

@st.cache_data(ttl=CACHE_TTL)
def filter_data_by_bounds(
    df: pd.DataFrame,
    bounds: Tuple[float, float, float, float],
    start_time: datetime,
    end_time: datetime,
    zoom_level: int = DEFAULT_ZOOM
) -> pd.DataFrame:
    """
    Filter data based on viewport bounds and time range
    """
    if df.empty:
        return df
    
    try:
        logger.info(f"Filtering data for time range: {start_time} to {end_time}")
        mask = (
            (df['latitude'].between(bounds[0], bounds[2])) &
            (df['longitude'].between(bounds[1], bounds[3])) &
            (df['timestamp'].between(start_time, end_time))
        )
        
        filtered_df = df[mask].copy()  # Create a copy to avoid SettingWithCopyWarning
        logger.info(f"Filtered data size: {len(filtered_df)} records")
        
        # Aggregate data based on zoom level
        filtered_df = aggregate_data(filtered_df, zoom_level)
        logger.info(f"Aggregated data size: {len(filtered_df)} records")
        
        # Ensure timestamp column is preserved for animation
        if 'timestamp' not in filtered_df.columns and 'utc_timestamp' in df.columns:
            # Get timestamp from original data for these points
            min_lat = filtered_df['latitude'].min()
            max_lat = filtered_df['latitude'].max()
            min_lon = filtered_df['longitude'].min()
            max_lon = filtered_df['longitude'].max()
            
            # Get timestamp values from the original dataframe
            timestamp_df = df[['latitude', 'longitude', 'timestamp']].copy()
            
            # Find closest points and assign timestamps
            logger.info("Preserving timestamp information for aggregated data")
            filtered_df = pd.merge_asof(
                filtered_df.sort_values('latitude'),
                timestamp_df.sort_values('latitude'),
                on='latitude',
                direction='nearest',
                suffixes=('', '_orig')
            )
            
            # Clean up columns
            if 'longitude_orig' in filtered_df.columns:
                filtered_df = filtered_df.drop('longitude_orig', axis=1)
        
        # Sample data if it's too large
        sampled_df = sample_data(filtered_df)
        logger.info(f"Final data size after sampling: {len(sampled_df)} records")
        
        return sampled_df
    except Exception as e:
        logger.error(f"Error filtering data: {e}")
        return pd.DataFrame()

def create_heatmap_layer(
    df: pd.DataFrame,
    weight_col: str = 'count',
    radius: int = 15,
    blur: int = 10
) -> List[List[float]]:
    """
    Create heatmap layer data
    """
    if df.empty:
        return []
    
    logger.info("Creating heatmap layer")
    return df[[
        'latitude',
        'longitude',
        weight_col
    ]].values.tolist()

def create_cluster_layer(
    df: pd.DataFrame,
    popup_col: str = 'count'
) -> List[Dict]:
    """
    Create cluster layer data
    """
    if df.empty:
        return []
    
    logger.info("Creating cluster layer")
    return [
        {
            'location': [row['latitude'], row['longitude']],
            'popup': f"Count: {row[popup_col]}"
        }
        for _, row in df.iterrows()
    ]

def get_center_and_zoom(df: pd.DataFrame) -> Tuple[List[float], int]:
    """
    Calculate the center coordinates and appropriate zoom level based on data bounds
    """
    if df.empty:
        return DEFAULT_CENTER, DEFAULT_ZOOM
    
    min_lat = df['latitude'].min()
    max_lat = df['latitude'].max()
    min_lon = df['longitude'].min()
    max_lon = df['longitude'].max()
    
    center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    
    # Calculate appropriate zoom level based on data spread
    lat_spread = max_lat - min_lat
    lon_spread = max_lon - min_lon
    max_spread = max(lat_spread, lon_spread)
    
    # Approximate zoom level calculation
    zoom = int(np.log2(360 / max_spread)) + 1
    zoom = min(max(zoom, MIN_ZOOM), MAX_ZOOM)  # Ensure zoom is within bounds
    
    logger.info(f"Map center: {center}, zoom level: {zoom}")
    return center, zoom

def create_map(
    df: pd.DataFrame,
    center: List[float] = None,
    zoom: int = None,
    use_clustering: bool = True
) -> folium.Map:
    """
    Create an interactive map with the data
    """
    logger.info("Creating map visualization")
    
    # Get center and zoom from data if not provided
    if center is None or zoom is None:
        center, zoom = get_center_and_zoom(df)
    
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='CartoDB positron',
        max_zoom=MAX_ZOOM,
        min_zoom=MIN_ZOOM,
        control_scale=True  # Add scale control
    )
    
    if df.empty:
        return m
    
    if use_clustering:
        # Use FastMarkerCluster for better performance
        cluster_data = create_cluster_layer(df)
        logger.info(f"Created cluster layer with {len(cluster_data)} points")
        
        marker_cluster = FastMarkerCluster(
            data=cluster_data,
            name='Clusters',
            options={
                'maxClusterRadius': 80,
                'disableClusteringAtZoom': 16,
                'spiderfyOnMaxZoom': True,
                'showCoverageOnHover': False,
                'zoomToBoundsOnClick': True
            }
        ).add_to(m)
    else:
        # Add heatmap layer with optimized settings
        heatmap_data = create_heatmap_layer(df)
        logger.info(f"Created heatmap layer with {len(heatmap_data)} points")
        
        HeatMap(
            heatmap_data,
            radius=15,
            blur=10,
            max_zoom=13,
            min_opacity=0.3,
            gradient={
                0.2: 'blue',
                0.4: 'lime',
                0.6: 'yellow',
                0.8: 'red'
            }
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def create_time_series(df: pd.DataFrame) -> Any:
    """
    Create time series visualization
    """
    if df.empty:
        logger.warning("No data available for time series visualization")
        return None
    
    try:
        logger.info("Creating time series visualization")
        # Group by timestamp and count, with daily aggregation
        logger.info("Aggregating data by day")
        time_series = df.groupby(pd.Grouper(key='timestamp', freq='D'))['count'].sum().reset_index()
        logger.info(f"Created time series with {len(time_series)} points")
        
        fig = px.line(
            time_series,
            x='timestamp',
            y='count',
            title='Daily Mobility Count'
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Count",
            showlegend=False,
            height=300  # Reduced height for better layout
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating time series: {e}")
        return None

def interpolate_points(df: pd.DataFrame, time_intervals: List[datetime]) -> pd.DataFrame:
    """
    Interpolate movement data between time points to create smoother animations
    """
    if len(df) < 4:  # Need at least 4 points for proper interpolation
        logger.warning("Not enough points for interpolation, skipping")
        df['interpolated'] = False
        return df
    
    logger.info("Interpolating points for smoother animation")
    
    # Get unique coordinate pairs to avoid duplicate points
    coords_df = df[['timestamp', 'latitude', 'longitude']].drop_duplicates()
    
    # Need at least 4 unique points in different locations for interpolation
    if len(coords_df) < 4:
        logger.warning("Not enough unique coordinates for interpolation, skipping")
        df['interpolated'] = False
        return df
    
    try:
        # Convert to numpy arrays for interpolation
        times = np.array([(t - time_intervals[0]).total_seconds() for t in coords_df['timestamp']])
        lats = np.array(coords_df['latitude'])
        lons = np.array(coords_df['longitude'])
        
        # Add small random noise to points to avoid collinearity
        # This helps with triangulation when points are too aligned
        np.random.seed(42)  # For reproducibility
        lats = lats + np.random.normal(0, 0.00001, lats.shape)
        lons = lons + np.random.normal(0, 0.00001, lons.shape)
        
        # Check if all points lie on a line - if so, add more random noise
        if np.std(lats) < 0.0001 or np.std(lons) < 0.0001:
            logger.warning("Points nearly collinear, adding noise for interpolation")
            lats = lats + np.random.normal(0, 0.0001, lats.shape)
            lons = lons + np.random.normal(0, 0.0001, lons.shape)
        
        # Use RBF interpolation which is more robust for scattered data
        lat_interp = Rbf(times, lats, function='linear')
        lon_interp = Rbf(times, lons, function='linear')
        
        # Create a more dense time series
        interp_times = np.linspace(times.min(), times.max(), min(len(times) * 2, 100))
        
        # Interpolate positions
        interp_lats = lat_interp(interp_times)
        interp_lons = lon_interp(interp_times)
        
        # Create new dataframe with interpolated points
        interp_df = pd.DataFrame({
            'timestamp': [time_intervals[0] + timedelta(seconds=t) for t in interp_times],
            'latitude': interp_lats,
            'longitude': interp_lons,
            'count': 1,
            'interpolated': True
        })
        
        # Remove NaN values from interpolation
        interp_df = interp_df.dropna(subset=['latitude', 'longitude'])
        
        # Combine with original data, keeping track of what's interpolated
        df['interpolated'] = False
        combined_df = pd.concat([df, interp_df], ignore_index=True)
        combined_df = combined_df.sort_values('timestamp')
        
        logger.info(f"Added {len(interp_df)} interpolated points for animation")
        return combined_df
    
    except Exception as e:
        # If interpolation fails, return original data without interpolation
        logger.error(f"Interpolation failed: {e}. Using original data.")
        df['interpolated'] = False
        return df

def identify_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify movement trajectories based on spatial and temporal proximity
    """
    if len(df) < 2:
        logger.warning("Not enough points to identify trajectories")
        df['trajectory_id'] = 0
        return df
    
    logger.info("Identifying movement trajectories")
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Calculate time differences and distances between consecutive points
    df_shifted = df.shift()
    df_shifted.iloc[0] = df.iloc[0]  # Avoid NaN on first row
    
    # Convert to numpy for faster computation
    coords1 = np.array(df[['latitude', 'longitude']])
    coords2 = np.array(df_shifted[['latitude', 'longitude']])
    
    # Calculate Haversine distance (approximate distance on Earth's surface)
    R = 6371.0  # Earth radius in km
    lat1, lon1 = np.radians(coords1[:, 0]), np.radians(coords1[:, 1])
    lat2, lon2 = np.radians(coords2[:, 0]), np.radians(coords2[:, 1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distances = R * c
    
    # Calculate time differences in seconds
    time_diffs = (df['timestamp'] - df_shifted['timestamp']).dt.total_seconds()
    
    # Initialize trajectory IDs
    df['trajectory_id'] = -1
    
    # Time and distance thresholds for segmentation
    # If points are more than 60 minutes apart or 5km apart, consider them different trajectories
    time_threshold = 60 * 60  # 60 minutes in seconds
    distance_threshold = 5.0  # 5 kilometers
    
    # Identify break points where a new trajectory should start
    break_points = np.where((time_diffs > time_threshold) | (distances > distance_threshold))[0]
    
    # Assign trajectory IDs
    current_id = 0
    last_break = 0
    
    for break_point in break_points:
        df.loc[last_break:break_point-1, 'trajectory_id'] = current_id
        current_id += 1
        last_break = break_point
    
    # Assign ID to the last segment
    df.loc[last_break:, 'trajectory_id'] = current_id
    
    logger.info(f"Identified {current_id + 1} distinct trajectories")
    return df

def create_smooth_path(points: List[Tuple[float, float]], smoothness: int = 50) -> List[Tuple[float, float]]:
    """
    Create a smooth path between points using spline interpolation
    """
    if len(points) < 2:
        return points
    
    # Extract x and y coordinates
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    
    # Create parameter t (0 to 1)
    t = np.linspace(0, 1, len(points))
    
    try:
        # Create spline functions for lat and lon
        lat_spline = UnivariateSpline(t, lats, k=min(3, len(points)-1))
        lon_spline = UnivariateSpline(t, lons, k=min(3, len(points)-1))
        
        # Create a more dense set of points
        t_smooth = np.linspace(0, 1, smoothness)
        smooth_lats = lat_spline(t_smooth)
        smooth_lons = lon_spline(t_smooth)
        
        # Combine back into points
        smooth_points = list(zip(smooth_lats, smooth_lons))
        return smooth_points
    except:
        # Fallback to linear interpolation
        t_smooth = np.linspace(0, 1, smoothness)
        smooth_lats = np.interp(t_smooth, t, lats)
        smooth_lons = np.interp(t_smooth, t, lons)
        return list(zip(smooth_lats, smooth_lons))

def prepare_animation_data(df: pd.DataFrame) -> List[Dict]:
    """
    Prepare data for animated visualization with improved transitions
    """
    if df.empty:
        return []
    
    logger.info("Preparing animation data")
    
    # Ensure timestamp column exists
    if 'timestamp' not in df.columns:
        logger.warning("Timestamp column not found in dataframe")
        # Try to regenerate timestamp from utc_timestamp if available
        if 'utc_timestamp' in df.columns:
            logger.info("Generating timestamp from utc_timestamp")
            df['timestamp'] = pd.to_datetime(df['utc_timestamp'])
        else:
            logger.error("Cannot create animation without timestamp data")
            return []
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    try:
        # Create time windows using the lowercase 'h'
        df['time_window'] = pd.to_datetime(df['timestamp']).dt.floor(ANIMATION_INTERVAL)
        
        # Identify movement trajectories
        df = identify_trajectories(df)
        
        # Get list of time windows - use np.sort instead of .sort() method
        time_windows = np.sort(df['time_window'].unique())
        
        # Create empty list for features
        features = []
        
        # Process each trajectory separately
        for trajectory_id, trajectory_group in df.groupby('trajectory_id'):
            # Skip trajectories with too few points
            if len(trajectory_group) < 2:
                continue
                
            # Sort trajectory by timestamp
            trajectory_group = trajectory_group.sort_values('timestamp')
            
            # Get points for this trajectory
            points = list(zip(trajectory_group['latitude'], trajectory_group['longitude']))
            timestamps = trajectory_group['timestamp'].tolist()
            time_windows_traj = trajectory_group['time_window'].unique()
            
            # Create smooth paths between points
            smooth_points = create_smooth_path(points, INTERPOLATION_SMOOTHNESS)
            
            # Distribute interpolated points across time windows
            total_duration = (timestamps[-1] - timestamps[0]).total_seconds()
            if total_duration == 0:  # All points at same time
                # Assign all points to the same time window
                for point in smooth_points:
                    features.append({
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Point',
                            'coordinates': [point[1], point[0]]  # lon, lat for GeoJSON
                        },
                        'properties': {
                            'time': time_windows_traj[0].strftime('%Y-%m-%d %H:%M:%S'),
                            'style': {'color': 'red'},
                            'icon': 'circle',
                            'iconstyle': {
                                'fillColor': '#ff3300',
                                'fillOpacity': 0.7,
                                'stroke': 'true',
                                'radius': 6
                            },
                            'trajectory': int(trajectory_id)
                        }
                    })
            else:
                # Interpolate timestamps for smooth points
                point_times = []
                for i in range(len(smooth_points)):
                    # Normalize to 0-1
                    t = i / (len(smooth_points) - 1)
                    interp_time = timestamps[0] + timedelta(seconds=t * total_duration)
                    point_times.append(interp_time)
                
                # Group points by time window
                for time_window in time_windows_traj:
                    window_start = time_window
                    window_end = window_start + timedelta(hours=int(ANIMATION_INTERVAL[0]))
                    
                    # Get points in this time window
                    window_points = []
                    for i, time in enumerate(point_times):
                        if window_start <= time < window_end:
                            window_points.append(smooth_points[i])
                    
                    # Add trail effect - include a few previous points
                    if USE_TRAILS and len(window_points) > 0:
                        # Find index of first point in this window
                        first_point_idx = next((i for i, time in enumerate(point_times) 
                                              if window_start <= time < window_end), None)
                        if first_point_idx is not None:
                            # Add previous points for trail effect
                            trail_start = max(0, first_point_idx - MAX_PATH_LENGTH)
                            trail_points = smooth_points[trail_start:first_point_idx]
                            
                            # Create path feature for trail
                            if len(trail_points) > 0:
                                trail_coords = [[p[1], p[0]] for p in trail_points]  # lon, lat
                                features.append({
                                    'type': 'Feature',
                                    'geometry': {
                                        'type': 'LineString',
                                        'coordinates': trail_coords
                                    },
                                    'properties': {
                                        'time': time_window.strftime('%Y-%m-%d %H:%M:%S'),
                                        'style': {
                                            'color': '#ff9999',
                                            'weight': 3,
                                            'opacity': 0.6
                                        },
                                        'trajectory': int(trajectory_id)
                                    }
                                })
                    
                    # Create features for points in this window
                    for i, point in enumerate(window_points):
                        # Determine the color based on position in the trajectory (red->yellow gradient)
                        position = i / max(len(window_points) - 1, 1)  # Normalize to 0-1
                        color = to_hex(cm.YlOrRd(0.3 + position * 0.7))
                        
                        features.append({
                            'type': 'Feature',
                            'geometry': {
                                'type': 'Point',
                                'coordinates': [point[1], point[0]]  # lon, lat for GeoJSON
                            },
                            'properties': {
                                'time': time_window.strftime('%Y-%m-%d %H:%M:%S'),
                                'style': {'color': color},
                                'icon': 'circle',
                                'iconstyle': {
                                    'fillColor': color,
                                    'fillOpacity': 0.7,
                                    'stroke': 'true',
                                    'radius': 6
                                },
                                'trajectory': int(trajectory_id)
                            }
                        })
        
        logger.info(f"Created {len(features)} animation features")
        return features
    except Exception as e:
        logger.error(f"Error creating animation data: {e}")
        return []

def create_animated_map(
    df: pd.DataFrame,
    center: List[float],
    zoom: int
) -> folium.Map:
    """
    Create an animated map showing movement patterns over time
    """
    logger.info("Creating animated map")
    
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='CartoDB positron',
        max_zoom=MAX_ZOOM,
        min_zoom=MIN_ZOOM,
        control_scale=True
    )
    
    if df.empty:
        return m
    
    # Prepare animation data
    features = prepare_animation_data(df)
    
    if not features:
        logger.warning("No animation features created, using fallback visualization")
        # Add a simple marker at the center as fallback
        folium.Marker(
            location=center,
            popup="No animation data available",
            icon=folium.Icon(color='red')
        ).add_to(m)
        return m
    
    # Add TimestampedGeoJson layer with improved settings
    try:
        TimestampedGeoJson(
            {
                'type': 'FeatureCollection',
                'features': features
            },
            period=f"PT{ANIMATION_INTERVAL[0]}h",
            duration='PT30M',  # Show 30 minutes worth of data at once
            transition_time=TRANSITION_TIME,
            auto_play=True,
            loop=True,
            max_speed=5,
            add_last_point=True,
            date_options='YYYY-MM-DD HH:mm:ss',
            time_slider_drag_update=True,
            speed_slider=True
        ).add_to(m)
    except Exception as e:
        logger.error(f"Error creating animation layer: {e}")
        # Add a simple heatmap as fallback
        heatmap_data = create_heatmap_layer(df)
        HeatMap(
            heatmap_data,
            radius=15,
            blur=10,
            max_zoom=13,
            min_opacity=0.3
        ).add_to(m)
    
    # Add tile layers for context
    folium.TileLayer(
        'CartoDB dark_matter',
        name='Dark Mode',
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        'OpenStreetMap',
        name='Street Map',
        control=True
    ).add_to(m)
    
    # Add controls
    folium.LayerControl().add_to(m)
    
    return m

def main():
    st.title("City Mobility Data Visualization")
    st.markdown("""
    This application visualizes mobility data to help city administrators and citizens understand movement patterns 
    throughout the city. View hotspots, common routes, and analyze trends over time.
    """)
    
    # Sidebar controls
    st.sidebar.header("Visualization Controls")
    
    # Get available date range
    min_date, max_date = get_available_date_range()
    
    # Define a maximum area constraint to limit load
    st.sidebar.subheader("Area Selection")
    area_option = st.sidebar.selectbox(
        "Select Map Focus",
        ["Default View", "Zoom to Data Center", "Custom Bounds"],
        index=1  # Default to zooming to data center
    )
    
    # Custom bounds if selected
    custom_bounds = None
    if area_option == "Custom Bounds":
        col_lat, col_lon = st.sidebar.columns(2)
        with col_lat:
            min_lat = st.sidebar.number_input("Min Latitude", -90.0, 90.0, 37.7, step=0.1)
            max_lat = st.sidebar.number_input("Max Latitude", -90.0, 90.0, 37.8, step=0.1)
        with col_lon:
            min_lon = st.sidebar.number_input("Min Longitude", -180.0, 180.0, -122.5, step=0.1)
            max_lon = st.sidebar.number_input("Max Longitude", -180.0, 180.0, -122.4, step=0.1)
        custom_bounds = (min_lat, min_lon, max_lat, max_lon)
    
    # Time range selection
    st.sidebar.subheader("Time Range")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            min_date.date()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            min_date.date() + timedelta(days=1)  # Default to one day for better performance
        )
    
    # Visualization type
    viz_type = st.sidebar.radio(
        "Visualization Type",
        ["Heatmap", "Clusters", "Animation"]
    )
    
    # Animation controls if animation selected
    if viz_type == "Animation":
        st.sidebar.subheader("Animation Controls")
        time_window = st.sidebar.slider(
            "Time Window (hours)",
            min_value=1,
            max_value=24,
            value=4,  # Changed default to 4 hours
            help="Amount of time to show in each animation frame"
        )
        
        # Apply the selected time window
        global ANIMATION_INTERVAL
        ANIMATION_INTERVAL = f"{time_window}h"  # Using 'h' instead of 'H'
        
        # Additional animation settings
        st.sidebar.subheader("Animation Settings")
        
        interpolate = st.sidebar.checkbox(
            "Interpolate Movement",
            value=True,
            help="Fill gaps between data points for smoother animation"
        )
        
        global INTERPOLATE_GAPS
        INTERPOLATE_GAPS = interpolate
        
        transition_time = st.sidebar.slider(
            "Transition Time (ms)",
            min_value=100,
            max_value=2000,
            value=500,
            step=100,
            help="Time to transition between frames (lower = faster)"
        )
        
        global TRANSITION_TIME
        TRANSITION_TIME = transition_time
    
    # Convert date inputs to datetime
    start_time = datetime.combine(start_date, datetime.min.time())
    end_time = datetime.combine(end_date, datetime.max.time())
    
    # Load data for the selected date range
    df = load_data_for_range(start_time, end_time)
    
    if df.empty:
        st.warning("No data available for the selected time range")
        return
    
    # Determine bounds based on selection
    if area_option == "Custom Bounds" and custom_bounds is not None:
        bounds = custom_bounds
    elif area_option == "Default View":
        bounds = (-90, -180, 90, 180)  # Full world view
    else:  # Zoom to Data Center
        # Calculate a smaller view around the data center
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        lat_spread = max(0.05, df['latitude'].std() * 2)  # Min 0.05 degrees, ~5km
        lon_spread = max(0.05, df['longitude'].std() * 2)
        bounds = (
            center_lat - lat_spread,
            center_lon - lon_spread,
            center_lat + lat_spread,
            center_lon + lon_spread
        )
    
    # Filter data by bounds
    df = filter_data_by_bounds(
        df,
        bounds=bounds,
        start_time=start_time,
        end_time=end_time
    )
    
    if df.empty:
        st.warning("No data available for the selected time range and location")
        return
    
    # Create visualizations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Interactive Map")
        
        # Get center and zoom from data or bounds
        if area_option == "Custom Bounds" and custom_bounds is not None:
            center = [(custom_bounds[0] + custom_bounds[2]) / 2, 
                      (custom_bounds[1] + custom_bounds[3]) / 2]
            # Calculate zoom level from bounds
            lat_diff = abs(custom_bounds[2] - custom_bounds[0])
            lon_diff = abs(custom_bounds[3] - custom_bounds[1])
            max_diff = max(lat_diff, lon_diff)
            zoom = int(np.log2(360 / max_diff)) + 1
            zoom = min(max(zoom, MIN_ZOOM), MAX_ZOOM)
        else:
            center, zoom = get_center_and_zoom(df)
        
        # Create appropriate map based on visualization type
        if viz_type == "Animation":
            m = create_animated_map(df, center, zoom)
        else:
            m = create_map(
                df,
                center=center,
                zoom=zoom,
                use_clustering=(viz_type == "Clusters")
            )
        
        # Display map
        st_folium(
            m,
            use_container_width=True,
            key=f"main_map_{viz_type}_{area_option}"  # Unique key for each viz type and area
        )
        
        if viz_type == "Animation":
            st.info("""
            The animation shows movement patterns over time. Red points are actual data points, 
            while orange points are interpolated to create smoother transitions. Use the slider 
            at the bottom of the map to control playback.
            """)
    
    with col2:
        st.subheader("Time Series")
        fig = create_time_series(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        st.write(f"Total Records: {len(df)}")
        st.write(f"Average Count: {df['count'].mean():.2f}")
        st.write(f"Max Count: {df['count'].max():.2f}")
        
        # Display current bounds
        st.subheader("Map Information")
        st.write(f"Center: {center}")
        st.write(f"Zoom Level: {zoom}")
        
        # Add data bounds
        if not df.empty:
            st.write("Data Bounds:")
            st.write(f"Lat: [{df['latitude'].min():.4f}, {df['latitude'].max():.4f}]")
            st.write(f"Lon: [{df['longitude'].min():.4f}, {df['longitude'].max():.4f}]")
            
            if viz_type == "Animation":
                st.write("Time Range:")
                time_min = df['timestamp'].min()
                time_max = df['timestamp'].max()
                st.write(f"Start: {time_min}")
                st.write(f"End: {time_max}")
                st.write(f"Duration: {time_max - time_min}")

if __name__ == "__main__":
    main() 