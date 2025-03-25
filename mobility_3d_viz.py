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

# Load environment variables
load_dotenv('.env.local')

# AWS S3 Configuration
S3_BUCKET_URL = os.getenv('S3_BUCKET')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Parse S3 bucket name and prefix
parsed_url = urlparse(S3_BUCKET_URL)
S3_BUCKET_NAME = parsed_url.netloc
S3_PREFIX = parsed_url.path.strip('/')

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
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
                    folder = prefix["Prefix"].split('/')[-2]  # Get the last folder name
                    if folder.startswith('date='):
                        date_folders.append(folder)
        
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
                    if obj["Key"].endswith('.parquet'):
                        parquet_files.append(obj["Key"])
        
        return parquet_files
    except Exception as e:
        logger.error(f"Error listing S3 files in {date_folder}: {e}")
        return []

@st.cache_data(ttl=CACHE_TTL)
def load_data_for_range(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Load data from S3 for specific date range"""
    try:
        logger.info(f"Loading data for date range: {start_date.date()} to {end_date.date()}")
        
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
        
        logger.info(f"Found {len(filtered_folders)} date folders in range")
        
        # Process each date folder
        dfs = []
        for date_folder in filtered_folders:
            logger.info(f"Processing folder: {date_folder}")
            
            # Get list of parquet files in the folder
            parquet_files = list_s3_parquet_files(date_folder)
            
            # Sample files if there are too many
            if len(parquet_files) > MAX_FILES_PER_DATE:
                parquet_files = np.random.choice(parquet_files, MAX_FILES_PER_DATE, replace=False)
            
            logger.info(f"Processing {len(parquet_files)} files from {date_folder}")
            
            # Process each file
            for s3_key in parquet_files:
                try:
                    # Read parquet file from S3
                    response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                    parquet_buffer = io.BytesIO(response['Body'].read())
                    
                    # Read the file with sampling
                    table = pq.read_table(
                        parquet_buffer,
                        columns=['utc_timestamp', 'latitude', 'longitude', 'ad_id']
                    )
                    
                    # Convert to pandas and sample
                    file_df = table.to_pandas()
                    sampled_df = file_df.sample(frac=SAMPLE_RATE)
                    
                    # Add date from folder name
                    folder_date = date_folder.split('=')[1]
                    sampled_df['folder_date'] = folder_date
                    
                    dfs.append(sampled_df)
                    
                except Exception as e:
                    logger.error(f"Error processing file {s3_key}: {e}")
                    continue
        
        # Combine all dataframes
        if not dfs:
            logger.error("No data loaded from any folder")
            return pd.DataFrame()
        
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(df)} sampled records")
        
        # Convert columns
        df['timestamp'] = pd.to_datetime(df['utc_timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        
        # Convert lat/long to float
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # Drop rows with invalid lat/long
        df = df.dropna(subset=['latitude', 'longitude'])
        logger.info(f"After cleaning: {len(df)} valid records")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()

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

def create_trajectories(df: pd.DataFrame, max_trajectories: int = 500) -> pd.DataFrame:
    """Create trajectories based on user IDs and timestamps"""
    # Group by ad_id and sort by timestamp
    trajectory_dfs = []
    
    # Get unique ad_ids
    unique_ids = df['ad_id'].unique()
    
    # Sample if too many
    if len(unique_ids) > max_trajectories:
        unique_ids = np.random.choice(unique_ids, max_trajectories, replace=False)
    
    # Process each user
    for user_id in unique_ids:
        user_df = df[df['ad_id'] == user_id].sort_values('timestamp')
        
        # Need at least 2 points for a trajectory
        if len(user_df) < 2:
            continue
            
        # Create trajectories for this user
        for i in range(len(user_df) - 1):
            start_point = user_df.iloc[i]
            end_point = user_df.iloc[i+1]
            
            # Skip if points are too far in time (more than 6 hours)
            time_diff = (end_point['timestamp'] - start_point['timestamp']).total_seconds()
            if time_diff > 6 * 3600:
                continue
                
            # Calculate distance
            lat_diff = end_point['latitude'] - start_point['latitude']
            lon_diff = end_point['longitude'] - start_point['longitude']
            dist = np.sqrt(lat_diff**2 + lon_diff**2)
            
            # Skip if distance is too small or too large
            if dist < 0.001 or dist > 0.5:  # About 100m to 50km
                continue
                
            # Create trajectory
            trajectory = {
                'start_lat': start_point['latitude'],
                'start_lon': start_point['longitude'],
                'end_lat': end_point['latitude'],
                'end_lon': end_point['longitude'],
                'start_time': start_point['timestamp'],
                'end_time': end_point['timestamp'],
                'user_id': user_id,
                'hour': start_point['hour'],
                'dist': dist
            }
            trajectory_dfs.append(pd.DataFrame([trajectory]))
    
    if not trajectory_dfs:
        return pd.DataFrame()
        
    trajectories = pd.concat(trajectory_dfs, ignore_index=True)
    logger.info(f"Created {len(trajectories)} trajectories")
    
    # Normalize distances for color
    if not trajectories.empty:
        trajectories['dist_norm'] = trajectories['dist'] / trajectories['dist'].max()
    
    return trajectories

def create_hexmap_layer(df: pd.DataFrame) -> pdk.Layer:
    """Create 3D hexagon map layer"""
    return pdk.Layer(
        "HexagonLayer",
        data=df,
        get_position=["longitude", "latitude"],
        radius=100,
        elevation_scale=10,
        elevation_range=[0, 1000],
        pickable=True,
        extruded=True,
        coverage=1,
        auto_highlight=True,
    )

def create_arc_layer(trajectories: pd.DataFrame) -> pdk.Layer:
    """Create arc layer for mobility flows"""
    return pdk.Layer(
        "ArcLayer",
        data=trajectories,
        get_source_position=["start_lon", "start_lat"],
        get_target_position=["end_lon", "end_lat"],
        get_source_color=[255, 0, 0, 180],
        get_target_color=[0, 0, 255, 180],
        get_width="dist * 500",  # Scale based on distance
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
    """Create 3D grid layer for density visualization"""
    return pdk.Layer(
        "GridLayer",
        data=df,
        get_position=["longitude", "latitude"],
        cell_size=100,
        elevation_scale=4,
        pickable=True,
        extruded=True,
    )

def create_scatterplot_layer(df: pd.DataFrame) -> pdk.Layer:
    """Create 3D scatterplot layer for point visualization"""
    return pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["longitude", "latitude"],
        get_color=[255, 0, 0, 160],
        get_radius=50,
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
            'color': [255, int(255 * (1 - traj['dist_norm'])), 0]
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

def main():
    # Header section with minimalist design
    st.title("Mobility Insights")
    st.markdown("""
    <p style='color: #86868B; font-size: 1.1rem; margin-bottom: 2rem;'>
    Explore movement patterns across the city through beautiful 3D visualizations.
    </p>
    """, unsafe_allow_html=True)
    
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
    
    # Load data for the selected date range
    df = load_data_for_range(start_time, end_time)
    
    if df.empty:
        st.warning("No data available for the selected time range")
        return
    
    # Apply hour filter
    df = df[(df['hour'] >= hour_filter[0]) & (df['hour'] < hour_filter[1])]
    
    if df.empty:
        st.warning("No data available for the selected hours")
        return
    
    # Create trajectories
    trajectories = create_trajectories(df)
    
    # Get map center
    map_center = [df['latitude'].mean(), df['longitude'].mean()]
    
    # Create visualization based on selection
    viz_container = st.container()
    with viz_container:
        # Map container with subtle shadow and rounded corners
        st.markdown("""
        <div style='background-color: white; padding: 1rem; border-radius: 20px; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);'>
        """, unsafe_allow_html=True)
        
        if viz_type == "Arc Map" and not trajectories.empty:
            st.markdown(f"<h3 style='color: #1D1D1F; margin-bottom: 1rem;'>Movement Flows</h3>", unsafe_allow_html=True)
            # Arc map showing mobility flows
            layer = create_arc_layer(trajectories)
            
            deck = pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v10",  # Light theme for Apple-like aesthetic
                initial_view_state=pdk.ViewState(
                    latitude=map_center[0],
                    longitude=map_center[1],
                    zoom=11,
                    pitch=50,
                    bearing=0,
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
                st.metric("Unique Users", f"{trajectories['user_id'].nunique():,}")
            with stats_col3:
                st.metric("Avg Distance", f"{trajectories['dist'].mean():.2f} km")
        
        elif viz_type == "Flow Map" and not trajectories.empty:
            st.markdown(f"<h3 style='color: #1D1D1F; margin-bottom: 1rem;'>Real-time Movement</h3>", unsafe_allow_html=True)
            # Prepare flow data
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
            
            view_state = pdk.ViewState(
                latitude=map_center[0],
                longitude=map_center[1],
                zoom=11,
                pitch=45,
                bearing=0
            )
            
            # Create and display deck
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                map_style="mapbox://styles/mapbox/dark-v10",
            )
            
            st.pydeck_chart(r, use_container_width=True, height=800)
            
            # Show time lapse animation control
            st.subheader("Time Lapse Animation")
            time_anim = st.empty()
            play_button = st.button("Play Animation")
            
            if play_button:
                for i in range(100):
                    # Update the current_time attribute of the layer
                    layer.current_time = i / 10
                    
                    # Update the deck with the new layer
                    r.layers = [layer]
                    
                    # Render the updated deck
                    time_anim.pydeck_chart(r, use_container_width=True, height=800)
                    time.sleep(0.1)
        
        elif viz_type == "Hexagon Map":
            st.markdown(f"<h3 style='color: #1D1D1F; margin-bottom: 1rem;'>Population Density</h3>", unsafe_allow_html=True)
            # 3D Hexbin map
            layer = pdk.Layer(
                "HexagonLayer",
                data=df,
                get_position=["longitude", "latitude"],
                radius=100,
                elevation_scale=elevation_scale,
                elevation_range=[0, 1000],
                extruded=True,
                pickable=True,
                auto_highlight=True,
                colorRange=[[0, 255, 0, 160],  # Green for low density
                           [128, 255, 0, 160],
                           [255, 255, 0, 160],
                           [255, 128, 0, 160],
                           [255, 0, 0, 160]],  # Red for high density
                colorAggregation='SUM'
            )
            
            deck = pdk.Deck(
                map_style="mapbox://styles/mapbox/dark-v10",
                initial_view_state=pdk.ViewState(
                    latitude=map_center[0],
                    longitude=map_center[1],
                    zoom=11,
                    pitch=50,
                    bearing=0,
                ),
                layers=[layer],
                tooltip={"text": "Count: {elevationValue}"}
            )
            
            st.pydeck_chart(deck, use_container_width=True, height=800)
        
        elif viz_type == "Grid Map":
            st.markdown(f"<h3 style='color: #1D1D1F; margin-bottom: 1rem;'>Activity Zones</h3>", unsafe_allow_html=True)
            # 3D Grid map
            layer = pdk.Layer(
                "GridLayer",
                data=df,
                get_position=["longitude", "latitude"],
                cell_size=200,
                elevation_scale=elevation_scale,
                pickable=True,
                extruded=True,
            )
            
            deck = pdk.Deck(
                map_style="mapbox://styles/mapbox/dark-v10",
                initial_view_state=pdk.ViewState(
                    latitude=map_center[0],
                    longitude=map_center[1],
                    zoom=11,
                    pitch=50,
                    bearing=0,
                ),
                layers=[layer],
                tooltip={"text": "Count: {count}"}
            )
            
            st.pydeck_chart(deck, use_container_width=True, height=800)
        
        else:  # Point Map
            st.markdown(f"<h3 style='color: #1D1D1F; margin-bottom: 1rem;'>Location Points</h3>", unsafe_allow_html=True)
            # Simple scatterplot
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position=["longitude", "latitude"],
                get_color=[255, 0, 0, 160],
                get_radius=50,
                pickable=True,
            )
            
            deck = pdk.Deck(
                map_style="mapbox://styles/mapbox/dark-v10",
                initial_view_state=pdk.ViewState(
                    latitude=map_center[0],
                    longitude=map_center[1],
                    zoom=11,
                    pitch=50,
                    bearing=0,
                ),
                layers=[layer],
                tooltip={"text": "Lat: {latitude}, Lon: {longitude}"}
            )
            
            st.pydeck_chart(deck, use_container_width=True, height=800)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Data insights section
    st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <h3 style='color: #1D1D1F; margin-bottom: 1.5rem;'>Data Insights</h3>
    """, unsafe_allow_html=True)
    
    # Modern metrics display
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    with metrics_col1:
        st.metric(
            "Total Records",
            f"{len(df):,}",
            delta=f"{len(df) - len(df[df['hour'] < 12]):,} in PM"
        )
    with metrics_col2:
        st.metric(
            "Unique Users",
            f"{df['ad_id'].nunique():,}"
        )
    with metrics_col3:
        st.metric(
            "Peak Hour",
            f"{df.groupby('hour').size().idxmax():02d}:00",
            delta="Most Active"
        )
    with metrics_col4:
        st.metric(
            "Date Range",
            f"{(end_date - start_date).days + 1} Days",
            delta=f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d')}"
        )
    
    # Activity chart with modern styling
    st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <h4 style='color: #1D1D1F; margin-bottom: 1rem;'>24-Hour Activity Pattern</h4>
    """, unsafe_allow_html=True)
    
    hour_counts = df.groupby('hour').size().reset_index()
    hour_counts.columns = ['Hour', 'Count']
    
    # Format hours in 12-hour format with AM/PM
    hour_counts['Hour_Format'] = hour_counts['Hour'].apply(
        lambda x: f"{x if x < 12 else x-12 if x > 12 else 12} {'AM' if x < 12 else 'PM'}"
    )
    
    st.bar_chart(
        hour_counts.set_index('Hour_Format')['Count'],
        use_container_width=True,
        height=400
    )

if __name__ == "__main__":
    main() 