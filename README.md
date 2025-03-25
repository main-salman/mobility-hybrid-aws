# Mobility Data Visualization with AWS Integration

A modern web application for visualizing large-scale mobility data using Streamlit and PyDeck, with AWS S3 integration for efficient data storage and retrieval.

## Project Purpose

This project aims to provide an efficient and interactive way to visualize large-scale mobility data by:
- Leveraging AWS S3 for scalable data storage
- Using Parquet format for optimized data access
- Implementing efficient data loading and caching strategies
- Providing multiple visualization types for different analysis needs
- Supporting interactive data exploration and filtering

## Features

### Visualization Types
- **Arc Map**: Visualize movement patterns between locations
- **Flow Map**: Show traffic flow and density
- **Hexagon Map**: Aggregate data into hexagonal bins for density analysis
- **Grid Map**: Display data in a grid-based format
- **Point Map**: Show individual data points with customizable styling

### Interactive Controls
- Date range selection with dynamic data loading
- Time period filtering (Morning, Afternoon, Evening, Night)
- Visualization style selection
- Interactive map controls
- Real-time data insights

### Performance Optimizations
- Efficient data loading from S3
- Smart data aggregation
- Caching mechanisms
- Optimized rendering for large datasets

## Technical Stack

- **Frontend**: Streamlit
- **Visualization**: PyDeck
- **Data Processing**: Pandas, NumPy
- **Storage**: AWS S3
- **Data Format**: Parquet
- **Python Version**: 3.8+

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/main-salman/mobility-hybrid-aws.git
   cd mobility-hybrid-aws
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up AWS credentials:
   - Create a `.env.local` file in the project root
   - Add your AWS credentials:
     ```
     AWS_ACCESS_KEY_ID=your_access_key
     AWS_SECRET_ACCESS_KEY=your_secret_key
     AWS_DEFAULT_REGION=your_region
     S3_BUCKET=your_bucket_name
     ```

5. Run the application:
   ```bash
   streamlit run mobility_3d_viz.py
   ```

## Data Structure

The application expects data to be stored in AWS S3 with the following structure:
```
s3://your-bucket/
    date=YYYY-MM-DD/
        *.parquet
```

Each Parquet file should contain the following columns:
- `start_lat`: Starting latitude
- `start_lon`: Starting longitude
- `end_lat`: Ending latitude
- `end_lon`: Ending longitude
- `hour`: Hour of the day (0-23)
- Additional attributes for visualization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 