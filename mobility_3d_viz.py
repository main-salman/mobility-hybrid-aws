import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from datetime import datetime, timedelta
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import time
import matplotlib.cm as cm
from matplotlib.colors import to_hex
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from urllib.parse import urlparse
import io
import json

# Load environment variables
load_dotenv('.env.local')

# AWS S3 Configuration
S3_BUCKET_NAME = os.getenv('S3_BUCKET')
S3_PREFIX = os.getenv('S3_PREFIX', '')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CENTER = [43.7, -79.4]  # Toronto area (based on your data)
DEFAULT_ZOOM = 11
CACHE_TTL = 3600  # 1 hour in seconds
SAMPLE_RATE = 0.01  # Sample 1% of data from each file
MAX_FILES_PER_DATE = 5  # Max number of files to process per date folder
GOOGLE_MAPS_API_KEY = os.getenv('NEXT_PUBLIC_GOOGLE_MAPS_API_KEY')

# Map style configuration
MAP_STYLE = "mapbox://styles/mapbox/light-v9"  # Use a light style base map

# Custom JavaScript for Google Maps integration
GOOGLE_MAPS_SCRIPT = """
<script src="https://maps.googleapis.com/maps/api/js?key=%s"></script>
<script>
window.addEventListener('load', function() {
    const mapboxContainer = document.querySelector('.mapboxgl-map');
    if (!mapboxContainer) return;
    
    const googleMap = new google.maps.Map(mapboxContainer, {
        center: {lat: %f, lng: %f},
        zoom: %d,
        styles: [
            {
                featureType: "all",
                elementType: "labels",
                stylers: [{ visibility: "off" }]
            }
        ]
    });
});
</script>
""" % (GOOGLE_MAPS_API_KEY, DEFAULT_CENTER[0], DEFAULT_CENTER[1], DEFAULT_ZOOM)

# Set page config
st.set_page_config(
    page_title="Mobility Insights",
    page_icon="🌆",
    layout="wide"
)

# Custom CSS for Apple-inspired design
st.markdown("""
<style>
    /* Main font and colors */
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Headers */
    h1 {
        font-weight: 600 !important;
        font-size: 2.5rem !important;
        letter-spacing: -0.5px !important;
        color: #1D1D1F !important;
        margin-bottom: 1rem !important;
    }
    
    h2, h3 {
        font-weight: 500 !important;
        color: #1D1D1F !important;
        letter-spacing: -0.3px !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #F5F5F7 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #0071E3 !important;
        color: white !important;
        border: none !important;
        border-radius: 980px !important;
        padding: 0.5rem 1rem !important;
        font-size: 0.9rem !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button:hover {
        background-color: #0077ED !important;
        transform: scale(1.02) !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #1D1D1F !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #86868B !important;
    }
    
    /* Charts */
    [data-testid="stChart"] {
        background-color: white !important;
        border-radius: 20px !important;
        padding: 1rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }
</style>
""", unsafe_allow_html=True)

def get_date_from_folder(folder_name: str) -> datetime:
    """Extract date from folder name"""
    date_str = folder_name.split('=')[1]
    return datetime.strptime(date_str, '%Y-%m-%d')

@st.cache_data(ttl=CACHE_TTL)
def list_s3_date_folders() -> List[str]:
    """List all date folders in S3 bucket"""
    try:
        # List objects with the prefix and delimiter to get "folders"
        paginator = s3_client.get_paginator('list_objects_v2')
        result = paginator.paginate(
            Bucket=S3_BUCKET_NAME,
            Prefix=f"{S3_PREFIX}/",
            Delimiter='/'
        )
        
        date_folders = []
        for page in result:
            if "CommonPrefixes" in page:
                for prefix in page["CommonPrefixes"]:
                    folder = prefix["Prefix"].rstrip('/').split('/')[-1]  # Get the last folder name
                    if folder.startswith('date='):
                        date_folders.append(folder)
        
        logger.info(f"Found date folders: {date_folders}")
        return sorted(date_folders)
    except Exception as e:
        logger.error(f"Error listing S3 folders: {e}")
        return []

@st.cache_data(ttl=CACHE_TTL)
def list_s3_parquet_files(date_folder: str) -> List[str]:
    """List all parquet files in a specific date folder"""
    try:
        # List objects in the date folder
        paginator = s3_client.get_paginator('list_objects_v2')
        result = paginator.paginate(
            Bucket=S3_BUCKET_NAME,
            Prefix=f"{S3_PREFIX}/{date_folder}/"
        )
        
        parquet_files = []
        for page in result:
            if "Contents" in page:
                for obj in page["Contents"]:
                    if obj["Key"].endswith('.snappy.parquet'):  # Only get .snappy.parquet files
                        parquet_files.append(obj["Key"])
        
        logger.info(f"Found {len(parquet_files)} parquet files in {date_folder}")
        return parquet_files
    except Exception as e:
        logger.error(f"Error listing S3 files in {date_folder}: {e}")
        return []

@st.cache_data(ttl=CACHE_TTL)
def get_s3_data(date_folder, file_name):
    """Fetch data from S3 and return as JSON"""
    try:
        key = f"{S3_PREFIX}/{date_folder}/{file_name}"
        logger.info(f"Fetching S3 object: {key}")
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        df = pd.read_parquet(response['Body'])
        return df.to_json(orient='records')
    except Exception as e:
        logger.error(f"Error fetching data from S3: {str(e)}")
        return None

# Add API endpoint
if st.query_params.get('api') == ['s3_data']:
    date_folder = st.query_params.get('date', [''])[0]
    file_name = st.query_params.get('file', [''])[0]
    
    if date_folder and file_name:
        data = get_s3_data(date_folder, file_name)
        if data:
            st.json(data)
        else:
            st.error("Failed to fetch data from S3")
    else:
        st.error("Missing date or file parameters")
    st.stop()

def get_time_period_hours(period: str) -> Tuple[int, int]:
    """Convert time period to hour range"""
    periods = {
        "Night (0:00-4:00)": (0, 4),
        "Early Morning (4:00-7:00)": (4, 7),
        "Morning (7:00-11:00)": (7, 11),
        "Lunch (11:00-14:00)": (11, 14),
        "Afternoon (14:00-17:00)": (14, 17),
        "Evening (17:00-21:00)": (17, 21),
        "Late Night (21:00-24:00)": (21, 24),
        "All Day": (0, 24)
    }
    return periods.get(period, (0, 24))

def create_trajectories(df: pd.DataFrame, max_trajectories: int = 100) -> pd.DataFrame:
    """Create trajectories based on aggregated data"""
    # Group by hour and location to create flows
    trajectory_dfs = []
    
    # Get unique locations
    unique_locations = df[['latitude', 'longitude']].drop_duplicates()
    
    # Sample if too many
    if len(unique_locations) > max_trajectories:
        unique_locations = unique_locations.sample(n=max_trajectories)
    
    # Process each location
    for _, loc in unique_locations.iterrows():
        loc_df = df[
            (df['latitude'] == loc['latitude']) & 
            (df['longitude'] == loc['longitude'])
        ].sort_values('hour')
        
        # Need at least 2 hours for a trajectory
        if len(loc_df) < 2:
            continue
            
        # Create trajectories between consecutive hours
        for i in range(len(loc_df) - 1):
            start_point = loc_df.iloc[i]
            end_point = loc_df.iloc[i+1]
            
            # Create trajectory
            trajectory = {
                'start_lat': start_point['latitude'],
                'start_lon': start_point['longitude'],
                'end_lat': end_point['latitude'],
                'end_lon': end_point['longitude'],
                'start_time': start_point['hour'],
                'end_time': end_point['hour'],
                'count': min(start_point['count'], end_point['count']),
                'hour': start_point['hour']
            }
            trajectory_dfs.append(pd.DataFrame([trajectory]))
    
    if not trajectory_dfs:
        return pd.DataFrame()
        
    trajectories = pd.concat(trajectory_dfs, ignore_index=True)
    logger.info(f"Created {len(trajectories)} trajectories")
    
    # Normalize counts for color
    if not trajectories.empty:
        trajectories['count_norm'] = trajectories['count'] / trajectories['count'].max()
    
    return trajectories

def create_hexmap_layer(df: pd.DataFrame) -> pdk.Layer:
    """Create 3D hexagon map layer with aggregated data"""
    return pdk.Layer(
        "HexagonLayer",
        data=df,
        get_position=["longitude", "latitude"],
        get_weight="count",  # Use count for weight
        radius=100,
        elevation_scale=10,
        elevation_range=[0, 1000],
        pickable=True,
        extruded=True,
        coverage=1,
        auto_highlight=True,
    )

def create_arc_layer(trajectories: pd.DataFrame) -> pdk.Layer:
    """Create arc layer for mobility flows with aggregated data"""
    return pdk.Layer(
        "ArcLayer",
        data=trajectories,
        get_source_position=["start_lon", "start_lat"],
        get_target_position=["end_lon", "end_lat"],
        get_source_color=[255, 0, 0, 180],
        get_target_color=[0, 0, 255, 180],
        get_width="count * 2",  # Scale width by count
        get_tilt=15,
        get_height=0.25,
        pickable=True,
    )

def create_flow_map(trajectories: pd.DataFrame) -> pdk.Layer:
    """Create flow map with animated, directional lines"""
    return pdk.Layer(
        "TripsLayer",
        data=trajectories,
        get_path="path",
        get_color=[255, 0, 0, 180],
        width_min_pixels=2,
        width_scale=5,
        joint_rounded=True,
        cap_rounded=True,
        pickable=True,
        trail_length=180,
        current_time=0,
    )

def create_grid_layer(df: pd.DataFrame) -> pdk.Layer:
    """Create 3D grid layer with aggregated data"""
    return pdk.Layer(
        "GridLayer",
        data=df,
        get_position=["longitude", "latitude"],
        get_weight="count",  # Use count for weight
        cell_size=100,
        elevation_scale=4,
        pickable=True,
        extruded=True,
    )

def create_scatterplot_layer(df: pd.DataFrame) -> pdk.Layer:
    """Create 3D scatterplot layer with aggregated data"""
    return pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["longitude", "latitude"],
        get_fill_color=[255, 0, 0, 160],
        get_radius="count * 50",  # Scale radius by count
        pickable=True,
    )

def prepare_flow_data(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for flow map visualization"""
    if trajectories.empty:
        return pd.DataFrame()
        
    # Create path field for trip layer
    trajectory_records = []
    
    for _, traj in trajectories.iterrows():
        rec = {
            'path': [[traj['start_lon'], traj['start_lat']], [traj['end_lon'], traj['end_lat']]],
            'timestamps': [0, 1],  # Start and end timestamps (normalized)
            'color': [255, int(255 * (1 - traj['count_norm'])), 0]
        }
        trajectory_records.append(rec)
    
    return pd.DataFrame(trajectory_records)

def create_heatmap_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Create aggregated heatmap data by hour"""
    if df.empty:
        return pd.DataFrame()
        
    # Group by hour and location
    hour_groups = []
    
    for hour in range(24):
        hour_df = df[df['hour'] == hour].copy()
        if not hour_df.empty:
            hour_df['count'] = 1
            hour_groups.append(hour_df)
    
    if not hour_groups:
        return pd.DataFrame()
        
    return pd.concat(hour_groups)

@st.cache_data(ttl=CACHE_TTL)
def get_available_date_range() -> Tuple[datetime, datetime]:
    """Get available date range from S3 folder names"""
    try:
        date_folders = list_s3_date_folders()
        
        if not date_folders:
            logger.error("No date folders found in S3")
            return datetime.now(), datetime.now()
        
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

@st.cache_data(ttl=CACHE_TTL)
def get_s3_presigned_url(key: str) -> str:
    """Generate a presigned URL for an S3 object"""
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET_NAME, 'Key': key},
            ExpiresIn=3600  # URL valid for 1 hour
        )
        return url
    except Exception as e:
        logger.error(f"Error generating presigned URL for {key}: {e}")
        return None

def format_size(size_bytes):
    """Format bytes into human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def get_viewport_bounds(deck: pdk.Deck) -> Tuple[float, float, float, float]:
    """Get the current viewport bounds from the deck"""
    view_state = deck.initial_view_state
    lat = view_state.latitude
    lon = view_state.longitude
    zoom = view_state.zoom
    
    # Calculate bounds based on zoom level with more granular control
    if zoom < 8:
        lat_delta = 1.0  # ~100km for very zoomed out
        lon_delta = 1.0
    elif zoom < 10:
        lat_delta = 0.5  # ~50km for zoomed out
        lon_delta = 0.5
    elif zoom < 12:
        lat_delta = 0.2  # ~20km for medium zoom
        lon_delta = 0.2
    elif zoom < 14:
        lat_delta = 0.1  # ~10km for zoomed in
        lon_delta = 0.1
    else:
        lat_delta = 0.05  # ~5km for very zoomed in
        lon_delta = 0.05
    
    # Ensure we're covering the Toronto area with more generous bounds
    min_lat = max(43.0, lat - lat_delta)
    max_lat = min(44.0, lat + lat_delta)
    min_lon = max(-80.0, lon - lon_delta)
    max_lon = min(-79.0, lon + lon_delta)
    
    return (min_lat, max_lat, min_lon, max_lon)

def get_resolution_for_zoom(zoom: float) -> float:
    """Get appropriate resolution based on zoom level"""
    if zoom < 8:
        return 0.1  # ~10km resolution for very zoomed out
    elif zoom < 10:
        return 0.05  # ~5km resolution for zoomed out
    elif zoom < 12:
        return 0.02  # ~2km resolution for medium zoom
    elif zoom < 14:
        return 0.01  # ~1km resolution for zoomed in
    else:
        return 0.005  # ~500m resolution for very zoomed in

@st.cache_data(ttl=CACHE_TTL)
def load_data_for_date_range(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Load data for entire date range - simplified approach"""
    try:
        logger.info(f"Loading data for date range: {start_date.date()} to {end_date.date()}")
        
        # Initialize metrics
        total_bytes_transferred = 0
        total_files_processed = 0
        total_files_failed = 0
        start_time = time.time()
        
        # Get list of date folders from S3
        date_folders = list_s3_date_folders()
        
        # Filter folders by date range
        filtered_folders = []
        for folder in date_folders:
            folder_date = get_date_from_folder(folder)
            if start_date.date() <= folder_date.date() <= end_date.date():
                filtered_folders.append(folder)
        
        if not filtered_folders:
            logger.warning(f"No date folders found in range: {start_date.date()} to {end_date.date()}")
            return pd.DataFrame()
        
        # Load data from each folder
        dfs = []
        
        for date_folder in filtered_folders:
            parquet_files = list_s3_parquet_files(date_folder)
            
            # Sort files by size (largest first) to prioritize files with more data
            file_sizes = []
            for s3_key in parquet_files:
                try:
                    response = s3_client.head_object(
                        Bucket=S3_BUCKET_NAME,
                        Key=s3_key
                    )
                    file_sizes.append((s3_key, response['ContentLength']))
                except Exception as e:
                    logger.error(f"Error getting file size for {s3_key}: {e}")
                    continue
            
            # Sort by size, largest first
            file_sizes.sort(key=lambda x: x[1], reverse=True)
            
            # Process up to 5 largest files per date
            for s3_key, file_size in file_sizes[:5]:
                try:
                    logger.info(f"Processing file: {s3_key} (size: {format_size(file_size)})")
                    
                    response = s3_client.get_object(
                        Bucket=S3_BUCKET_NAME,
                        Key=s3_key
                    )
                    
                    # Track file size
                    total_bytes_transferred += file_size
                    total_files_processed += 1
                    
                    # Read parquet file directly from S3 response
                    df = pd.read_parquet(io.BytesIO(response['Body'].read()))
                    logger.info(f"Loaded {len(df)} records from file")
                    
                    if len(df) == 0:
                        logger.warning(f"Empty dataframe from file: {s3_key}")
                        continue
                    
                    # For small files (less than 100 records), keep all data
                    # For larger files, sample based on size
                    if len(df) < 100:
                        sampled_df = df
                        logger.info("Small file, keeping all records")
                    else:
                        # Sample rate based on file size
                        sample_rate = min(1.0, max(0.2, 10000 / len(df)))
                        sampled_df = df.sample(frac=sample_rate)
                        logger.info(f"Sampled {len(sampled_df)} records (rate: {sample_rate:.2%})")
                    
                    # Add date from folder name
                    sampled_df['folder_date'] = date_folder.split('=')[1]
                    
                    # Filter out invalid coordinates
                    sampled_df['latitude'] = pd.to_numeric(sampled_df['latitude'], errors='coerce')
                    sampled_df['longitude'] = pd.to_numeric(sampled_df['longitude'], errors='coerce')
                    sampled_df = sampled_df.dropna(subset=['latitude', 'longitude'])
                    
                    # Filter to Toronto area with more generous bounds
                    sampled_df = sampled_df[
                        (sampled_df['latitude'].between(42.5, 44.5)) &  # Extended bounds
                        (sampled_df['longitude'].between(-80.5, -78.5))  # Extended bounds
                    ]
                    
                    if not sampled_df.empty:
                        dfs.append(sampled_df)
                        logger.info(f"Added {len(sampled_df)} records to dataset")
                    
                except Exception as e:
                    logger.error(f"Error processing file {s3_key}: {e}")
                    total_files_failed += 1
                    continue
        
        if not dfs:
            logger.error("No data loaded from any folder")
            return pd.DataFrame()
        
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(df)} total records")
        
        # Convert timestamp and extract hour
        df['timestamp'] = pd.to_datetime(df['utc_timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        
        # Aggregate data by location and hour
        df['lat_rounded'] = df['latitude'].round(4)  # ~100m resolution
        df['lon_rounded'] = df['longitude'].round(4)
        
        # Group by location and hour, count occurrences
        df = df.groupby(['lat_rounded', 'lon_rounded', 'hour']).agg({
            'ad_id': 'count',
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()
        
        # Rename columns
        df = df.rename(columns={'ad_id': 'count'})
        
        logger.info(f"After aggregation: {len(df)} unique locations")
        
        # Calculate loading time
        loading_time = time.time() - start_time
        
        # Store transfer metrics in session state
        st.session_state['data_transfer_metrics'] = {
            'total_bytes': total_bytes_transferred,
            'files_processed': total_files_processed,
            'files_failed': total_files_failed,
            'formatted_size': format_size(total_bytes_transferred),
            'loading_time': f"{loading_time:.2f}s",
            'points_displayed': len(df)
        }
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()

def update_viewport(view_state: dict):
    """Update viewport state and trigger data reload"""
    # Store the new view state in session state
    st.session_state.viewport_state = view_state
    
    # Force a rerun to load new data
    st.rerun()

def main():
    # Header section with minimalist design
    st.title("Mobility Insights")
    st.markdown("""
    <p style='color: #86868B; font-size: 1.1rem; margin-bottom: 2rem;'>
    Explore movement patterns across the city through beautiful 3D visualizations.
    </p>
    """ + GOOGLE_MAPS_SCRIPT, unsafe_allow_html=True)
    
    # Sidebar with clean design
    with st.sidebar:
        st.markdown("### Visualization Settings")
        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
        
        # Get available date range
        min_date, max_date = get_available_date_range()
        
        # Date selection with available dates
        st.markdown("#### Select Date Range")
        available_dates = []
        date_folders = list_s3_date_folders()
        
        for folder in date_folders:
            try:
                date_str = folder.split('=')[1]
                date = datetime.strptime(date_str, '%Y-%m-%d').date()
                available_dates.append(date)
            except Exception as e:
                logger.error(f"Error parsing date from folder {folder}: {e}")
                continue
        
        if not available_dates:
            st.error("No data available in S3 bucket")
            return
        
        available_dates.sort()
        
        # Create date selection with only available dates
        start_date = st.selectbox(
            "Start Date",
            options=available_dates,
            index=0,
            format_func=lambda x: x.strftime("%B %d, %Y"),
            help="Select from available dates"
        )
        
        end_date = st.selectbox(
            "End Date",
            options=[d for d in available_dates if d >= start_date],
            index=0,
            format_func=lambda x: x.strftime("%B %d, %Y"),
            help="Select from available dates"
        )
        
        st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
        
        # Time period selection with modern buttons
        st.markdown("#### Time of Day")
        
        # Create 2 rows of 4 columns for time period buttons
        col1, col2, col3, col4 = st.columns(4)
        col5, col6, col7, col8 = st.columns(4)
        
        # Dictionary to store button states with cleaner labels
        time_buttons = {
            "All Day": col1.button("All Day", use_container_width=True),
            "Night": col2.button("Night", help="12 AM - 4 AM", use_container_width=True),
            "Early Morning": col3.button("Dawn", help="4 AM - 7 AM", use_container_width=True),
            "Morning": col4.button("Morning", help="7 AM - 11 AM", use_container_width=True),
            "Lunch": col5.button("Lunch", help="11 AM - 2 PM", use_container_width=True),
            "Afternoon": col6.button("Afternoon", help="2 PM - 5 PM", use_container_width=True),
            "Evening": col7.button("Evening", help="5 PM - 9 PM", use_container_width=True),
            "Late Night": col8.button("Night", help="9 PM - 12 AM", use_container_width=True)
        }
        
        st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
        
        # Find which button was clicked (default to "All Day" if none clicked)
        selected_period = "All Day"
        for period, clicked in time_buttons.items():
            if clicked:
                selected_period = period
                break
        
        # Map the selected period to the format expected by get_time_period_hours
        period_mapping = {
            "All Day": "All Day",
            "Night": "Night (0:00-4:00)",
            "Early Morning": "Early Morning (4:00-7:00)",
            "Morning": "Morning (7:00-11:00)",
            "Lunch": "Lunch (11:00-14:00)",
            "Afternoon": "Afternoon (14:00-17:00)",
            "Evening": "Evening (17:00-21:00)",
            "Late Night": "Late Night (21:00-24:00)"
        }
        
        # Get hour range from time period
        hour_filter = get_time_period_hours(period_mapping[selected_period])
        
        # Visualization type with modern radio buttons
        st.markdown("#### Visualization Style")
        viz_type = st.radio(
            "",
            ["Arc Map", "Flow Map", "Hexagon Map", "Grid Map", "Point Map"],
            format_func=lambda x: x.replace(" Map", "")
        )
        
        st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
        
        # Elevation scale with modern slider
        st.markdown("#### Elevation Scale")
        elevation_scale = st.slider(
            "",
            min_value=1,
            max_value=20,
            value=4,
            help="Adjust the height of 3D elements"
        )
    
    # Convert date inputs to datetime
    start_time = datetime.combine(start_date, datetime.min.time())
    end_time = datetime.combine(end_date, datetime.max.time())
    
    # Initialize viewport state if not exists
    if 'viewport_state' not in st.session_state:
        st.session_state.viewport_state = {
            'latitude': DEFAULT_CENTER[0],
            'longitude': DEFAULT_CENTER[1],
            'zoom': DEFAULT_ZOOM,
            'pitch': 50,
            'bearing': 0
        }
    
    # Create a key for the pydeck chart that includes the viewport state
    viewport_key = f"viewport_{st.session_state.viewport_state['latitude']}_{st.session_state.viewport_state['longitude']}_{st.session_state.viewport_state['zoom']}"
    
    # Get current viewport bounds and resolution
    deck = pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=st.session_state.viewport_state['latitude'],
            longitude=st.session_state.viewport_state['longitude'],
            zoom=st.session_state.viewport_state['zoom'],
            pitch=st.session_state.viewport_state['pitch'],
            bearing=st.session_state.viewport_state['bearing']
        )
    )
    
    viewport_bounds = get_viewport_bounds(deck)
    resolution = get_resolution_for_zoom(st.session_state.viewport_state['zoom'])
    
    # Load data for the entire date range
    with st.spinner("Loading data..."):
        df = load_data_for_date_range(
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.max.time())
        )
    
    # Display metrics
    if 'data_transfer_metrics' in st.session_state:
        metrics = st.session_state['data_transfer_metrics']
        st.info(f"""
            Data Transfer Metrics:
            - Size: {metrics['formatted_size']}
            - Files Processed: {metrics['files_processed']}
            - Files Failed: {metrics['files_failed']}
            - Loading Time: {metrics['loading_time']}
            - Points Displayed: {metrics['points_displayed']}
        """)
    
    if df.empty:
        st.warning("No data available for the selected date range.")
        return
    
    # Apply hour filter
    df = df[(df['hour'] >= hour_filter[0]) & (df['hour'] < hour_filter[1])]
    
    if df.empty:
        st.warning("No data available for the selected hours. Please try a different time period.")
        return
    
    # Create trajectories
    trajectories = create_trajectories(df)
    
    # Create visualization based on selection
    viz_container = st.container()
    with viz_container:
        st.markdown("""
        <div style='background-color: white; padding: 1rem; border-radius: 20px; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);'>
        """, unsafe_allow_html=True)
        
        # Create visualization based on selection
        if viz_type == "Arc Map" and not trajectories.empty:
            st.markdown(f"<h3 style='color: #1D1D1F; margin-bottom: 1rem;'>Movement Flows</h3>", unsafe_allow_html=True)
            layer = create_arc_layer(trajectories)
            
            deck = pdk.Deck(
                map_style=MAP_STYLE,
                initial_view_state=pdk.ViewState(
                    latitude=DEFAULT_CENTER[0],
                    longitude=DEFAULT_CENTER[1],
                    zoom=DEFAULT_ZOOM,
                    pitch=50,
                    bearing=0
                ),
                layers=[layer],
                tooltip={"text": "Start: {start_lat}, {start_lon}\nEnd: {end_lat}, {end_lon}"}
            )
            
            st.pydeck_chart(deck, use_container_width=True, height=800)
            
            # Stats in a modern grid
            st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1:
                st.metric("Total Flows", f"{len(trajectories):,}")
            with stats_col2:
                st.metric("Total Count", f"{trajectories['count'].sum():,}")
            with stats_col3:
                st.metric("Avg Count", f"{trajectories['count'].mean():.1f}")
        
        elif viz_type == "Flow Map" and not trajectories.empty:
            st.markdown(f"<h3 style='color: #1D1D1F; margin-bottom: 1rem;'>Real-time Movement</h3>", unsafe_allow_html=True)
            flow_data = prepare_flow_data(trajectories)
            
            layer = pdk.Layer(
                "TripsLayer",
                flow_data,
                get_path="path",
                get_color="color",
                width_min_pixels=3,
                rounded=True,
                trail_length=0.5,
                current_time=0
            )
            
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=pdk.ViewState(
                    latitude=DEFAULT_CENTER[0],
                    longitude=DEFAULT_CENTER[1],
                    zoom=DEFAULT_ZOOM,
                    pitch=50,
                    bearing=0
                ),
                map_style=MAP_STYLE
            )
            
            st.pydeck_chart(deck, use_container_width=True, height=800)
        
        elif viz_type == "Hexagon Map":
            st.markdown(f"<h3 style='color: #1D1D1F; margin-bottom: 1rem;'>Population Density</h3>", unsafe_allow_html=True)
            layer = pdk.Layer(
                "HexagonLayer",
                data=df,
                get_position=["longitude", "latitude"],
                get_weight="count",
                radius=100,
                elevation_scale=elevation_scale,
                elevation_range=[0, 1000],
                extruded=True,
                pickable=True,
                auto_highlight=True,
                colorRange=[[0, 255, 0, 160],
                           [128, 255, 0, 160],
                           [255, 255, 0, 160],
                           [255, 128, 0, 160],
                           [255, 0, 0, 160]],
                colorAggregation='SUM'
            )
            
            deck = pdk.Deck(
                map_style=MAP_STYLE,
                initial_view_state=pdk.ViewState(
                    latitude=DEFAULT_CENTER[0],
                    longitude=DEFAULT_CENTER[1],
                    zoom=DEFAULT_ZOOM,
                    pitch=50,
                    bearing=0
                ),
                layers=[layer],
                tooltip={"text": "Count: {elevationValue}"}
            )
            
            st.pydeck_chart(deck, use_container_width=True, height=800)
        
        elif viz_type == "Grid Map":
            st.markdown(f"<h3 style='color: #1D1D1F; margin-bottom: 1rem;'>Activity Zones</h3>", unsafe_allow_html=True)
            layer = pdk.Layer(
                "GridLayer",
                data=df,
                get_position=["longitude", "latitude"],
                get_weight="count",
                cell_size=200,
                elevation_scale=elevation_scale,
                pickable=True,
                extruded=True,
            )
            
            deck = pdk.Deck(
                map_style=MAP_STYLE,
                initial_view_state=pdk.ViewState(
                    latitude=DEFAULT_CENTER[0],
                    longitude=DEFAULT_CENTER[1],
                    zoom=DEFAULT_ZOOM,
                    pitch=50,
                    bearing=0
                ),
                layers=[layer],
                tooltip={"text": "Count: {count}"}
            )
            
            st.pydeck_chart(deck, use_container_width=True, height=800)
        
        else:  # Point Map
            st.markdown(f"<h3 style='color: #1D1D1F; margin-bottom: 1rem;'>Location Points</h3>", unsafe_allow_html=True)
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position=["longitude", "latitude"],
                get_fill_color=[255, 0, 0, 160],
                get_radius="count * 50",
                pickable=True,
            )
            
            deck = pdk.Deck(
                map_style=MAP_STYLE,
                initial_view_state=pdk.ViewState(
                    latitude=DEFAULT_CENTER[0],
                    longitude=DEFAULT_CENTER[1],
                    zoom=DEFAULT_ZOOM,
                    pitch=50,
                    bearing=0
                ),
                layers=[layer],
                tooltip={"text": "Lat: {latitude}, Lon: {longitude}"}
            )
            
            st.pydeck_chart(deck, use_container_width=True, height=800)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Add JavaScript to handle viewport changes
    st.markdown("""
    <script>
    // Listen for viewport changes from pydeck
    window.addEventListener('message', (event) => {
        if (event.data.type === 'pydeck_view_state_change') {
            const viewState = event.data.viewState;
            
            // Send the viewport state to Streamlit
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: {
                    viewport: {
                        latitude: viewState.latitude,
                        longitude: viewState.longitude,
                        zoom: viewState.zoom,
                        pitch: viewState.pitch,
                        bearing: viewState.bearing
                    }
                }
            }, '*');
        }
    });
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 