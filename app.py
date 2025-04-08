import sys
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import open3d as o3d
from pathlib import Path
import subprocess
import glob
from PIL import Image
import io
import base64
import plotly.graph_objects as go

# def check_environment():
#     env_status = {}
#     env_status["base_dir"] = os.path.exists(BASE_DIR)
#     env_status["sfm_dir"] = os.path.exists(SFM_DIR)
#     env_status["data_dir"] = os.path.exists(DATA_DIR)
#     env_status["script_dir"] = os.path.exists(SFM_DIR / "script")
#     env_status["temp_writable"] = os.access(TEMP_DIR, os.W_OK)
#     return env_status

# def get_environment_info():
    # """Get information about the execution environment for debugging"""
    # info = {
    #     "python_version": sys.version,
    #     "current_directory": os.getcwd(),
    #     "path_exists": {
    #         "BASE_DIR": os.path.exists(BASE_DIR),
    #         "SFM_DIR": os.path.exists(SFM_DIR),
    #         "DATA_DIR": os.path.exists(DATA_DIR),
    #         "TEMP_DIR": os.path.exists(TEMP_DIR),
    #         "script_dir": os.path.exists(SFM_DIR / "script")
    #     },
    #     "directory_contents": {
    #         "SFM/script": os.listdir(SFM_DIR / "script") if os.path.exists(SFM_DIR / "script") else "Not found",
    #         "SFM/data": os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else "Not found"
    #     }
    # }
    # return info

# Set page configuration
st.set_page_config(
    page_title="3D Reconstruction Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("3D Reconstruction with Uncalibrated Stereo")
st.markdown("""
This application demonstrates 3D reconstruction using uncalibrated stereo vision techniques.
Upload your own stereo images or use our provided examples to see the reconstruction pipeline in action.
""")

# Create sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload Images", "View Results", "About"])

# Define paths
DATA_DIR = Path("Data")
OUTPUT_DIR = DATA_DIR / "Output"
TEMPLE_DIR = DATA_DIR / "temple" / "undistorted"
ROCK_DIR = DATA_DIR / "rock" / "undistorted"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def display_point_cloud(ply_path, width=800, height=600):
    """Display a point cloud from a PLY file using Plotly."""
    # Check if file exists
    if not os.path.exists(ply_path):
        st.error(f"PLY file not found at {ply_path}")
        return
    
    try:
        # Try to read the point cloud
        points = None
        
        if HAS_O3D:
            # Use Open3D if available
            try:
                pcd = o3d.io.read_point_cloud(ply_path)
                points = np.asarray(pcd.points)
            except Exception as e:
                st.warning(f"Error using Open3D to read PLY: {e}")
                # Fall back to manual parsing
                points = None
        
        # If Open3D failed or isn't available, parse the PLY file manually
        if points is None:
            points = []
            with open(ply_path, 'r') as f:
                lines = f.readlines()
                
            # Skip header
            header_end = 0
            for i, line in enumerate(lines):
                if "end_header" in line:
                    header_end = i + 1
                    break
            
            # Parse points
            for i in range(header_end, len(lines)):
                if lines[i].strip():
                    parts = lines[i].strip().split()
                    if len(parts) >= 3:
                        try:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            points.append([x, y, z])
                        except ValueError:
                            continue
            
            # Convert to numpy array
            points = np.array(points)
        
        if len(points) == 0:
            st.error("No valid points found in the PLY file")
            return
        
        # Downsample points for better performance if needed
        max_points = 25000  # Adjust based on performance
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
        
        # Create a Plotly 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=points[:, 2],  # Color by Z coordinate
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(
                    title="Z Coordinate",
                    thickness=20
                )
            )
        )])
        
        # Update the layout for better visualization
        fig.update_layout(
            title=f"Point Cloud: {os.path.basename(ply_path)}",
            width=width,
            height=height,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'  # This keeps the aspect ratio true to the data
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Display the interactive 3D plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Add download button for the PLY file
        with open(ply_path, 'rb') as f:
            ply_data = f.read()
        
        st.download_button(
            label="Download PLY File",
            data=ply_data,
            file_name=os.path.basename(ply_path),
            mime="application/octet-stream"
        )
        
    except Exception as e:
        st.error(f"Error displaying point cloud: {e}")
        import traceback
        st.code(traceback.format_exc())

def run_reconstruction(dataset_choice, topology_choice):
    """Run the 3D reconstruction process using Temple.py or temple_icp_merger.py"""
    try:
        st.info("Starting 3D reconstruction process...")

        st.session_state.last_dataset = dataset_choice.lower()
        st.session_state.last_topology = topology_choice
        
        # Determine dataset path
        if dataset_choice == "Temple":
            dataset_path = TEMPLE_DIR
            dataset_type = "temple"
        else:  # Dino
            dataset_path = ROCK_DIR
            dataset_type = "dino"
            
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.dirname(os.__file__) + '/site-packages' + env.get('PYTHONPATH', '')
        # Run the reconstruction script with appropriate parameters
        cmd = [
            sys.executable, "Temple.py",
            "--input_dir", str(dataset_path),
            "--output_dir", str(OUTPUT_DIR),
            "--dataset_type", dataset_type,
            "--topology", topology_choice
        ]
        
        # Execute command with realtime output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Create a placeholder for the output
        output_placeholder = st.empty()
        complete_log = ""
        
        # Stream the output
        for line in process.stdout:
            complete_log += line
            output_placeholder.code(complete_log)
        
        # Wait for the process to complete
        process.wait()
        
        # Check if process was successful
        if process.returncode == 0:
            st.success("3D reconstruction completed successfully!")
            
            # Run ICP if requested
            if st.checkbox("Run ICP alignment on point clouds?", value=False):
                st.info("Running ICP alignment...")
                
                icp_cmd = [
                    "python", "temple_icp_merger.py",
                    "--input_dir", str(OUTPUT_DIR),
                    "--output_dir", str(OUTPUT_DIR / "Merged")
                ]
                
                icp_process = subprocess.Popen(
                    icp_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                
                icp_output = st.empty()
                icp_log = ""
                
                for line in icp_process.stdout:
                    icp_log += line
                    icp_output.code(icp_log)
                
                icp_process.wait()
                
                if icp_process.returncode == 0:
                    st.success("ICP alignment completed successfully!")
                else:
                    st.error("ICP alignment failed.")
        else:
            st.error("3D reconstruction failed.")
        
        return True
    except Exception as e:
        st.error(f"Error during reconstruction: {e}")
        return False

def display_image_grid(image_files, cols=3, width=None):
    """Display a grid of images using st.columns."""
    # Filter to only include image files
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        st.warning("No images found.")
        return
    
    # Calculate number of rows needed
    rows = (len(image_files) + cols - 1) // cols
    
    # Create grid
    for i in range(rows):
        # Create columns for this row
        columns = st.columns(cols)
        
        # Fill columns with images
        for j in range(cols):
            idx = i * cols + j
            if idx < len(image_files):
                img_path = image_files[idx]
                with columns[j]:
                    st.image(img_path, caption=os.path.basename(img_path), width=width)

# HOME PAGE
if page == "Home":
    st.header("Welcome to 3D Reconstruction Demo")
    
    # Add two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### What is 3D Reconstruction?
        3D reconstruction is the process of capturing the shape and appearance of real objects
        using multiple 2D images taken from different viewpoints.
        
        This demo uses stereo vision techniques to:
        - Calibrate cameras
        - Rectify stereo image pairs
        - Compute disparity maps
        - Generate 3D point clouds
        - Optionally align multiple point clouds
        """)
    
    with col2:
        # Display sample result
        sample_image_path = os.path.join(str(OUTPUT_DIR), "pointcloud_perspective.png")
        if os.path.exists(sample_image_path):
            st.image(sample_image_path, caption="Sample 3D Reconstruction", width=350)
        else:
            st.info("Run a reconstruction to see results here.")
    
    # Quick start section
    st.header("Quick Start")
    
    # Create columns for the quick start options
    quick_col1, quick_col2, quick_col3 = st.columns([1, 1, 1])
    
    with quick_col1:
        dataset_choice = st.selectbox("Select Dataset", ["Temple", "Dino"])
    
    with quick_col2:
        topology_choice = st.selectbox(
            "Select Camera Topology", 
            ["skipping_2", "skipping_1", "adjacent", "overlapping", "360"]
        )
    
    with quick_col3:
        if st.button("Run Reconstruction"):
            success = run_reconstruction(dataset_choice, topology_choice)
            if success:
                st.info("Go to 'View Results' to see the reconstruction output.")

# UPLOAD IMAGES PAGE
elif page == "Upload Images":
    st.header("Upload Your Own Stereo Images")
    
    # Instructions
    st.markdown("""
    ### Instructions
    1. Upload a series of images taken around an object.
    2. The images should overlap significantly.
    3. Try to maintain consistent lighting.
    4. Avoid motion blur.
    """)
    
    # Image upload
    uploaded_files = st.file_uploader(
        "Upload Images", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} images.")
        
        # Create a custom directory for these uploads
        upload_dir = DATA_DIR / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the uploaded files
        saved_files = []
        for uploaded_file in uploaded_files:
            # Save the file
            file_path = upload_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(str(file_path))
        
        # Display the uploaded images
        st.subheader("Uploaded Images")
        display_image_grid(saved_files, width=150)
        
        # Run reconstruction on uploaded images
        st.subheader("Run Reconstruction")
        
        topology_choice = st.selectbox(
            "Select Camera Topology", 
            ["skipping_2", "skipping_1", "adjacent", "overlapping", "360"]
        )
        
        if st.button("Start Reconstruction"):
            cmd = [
                "python", "Temple.py",
                "--input_dir", str(upload_dir),
                "--output_dir", str(OUTPUT_DIR / "uploads"),
                "--topology", topology_choice
            ]
            
            st.info("Starting reconstruction process...")
            
            # Execute command
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Create a placeholder for the output
            output_placeholder = st.empty()
            complete_log = ""
            
            # Stream the output
            for line in process.stdout:
                complete_log += line
                output_placeholder.code(complete_log)
            
            # Wait for the process to complete
            process.wait()
            
            if process.returncode == 0:
                st.success("Reconstruction completed! Go to 'View Results' to see the output.")
            else:
                st.error("Reconstruction failed.")

# VIEW RESULTS PAGE
elif page == "View Results":
    st.header("View Reconstruction Results")
    
    # Get all output directories
    output_dirs = [d for d in os.listdir(str(OUTPUT_DIR)) if os.path.isdir(os.path.join(str(OUTPUT_DIR), d))]
    output_dirs = [""] + output_dirs  # Add root output dir
    
    # Initialize session state for tracking reconstruction parameters if not exists
    if 'last_dataset' not in st.session_state:
        st.session_state.last_dataset = "temple"
    if 'last_topology' not in st.session_state:
        st.session_state.last_topology = "skipping_2"
    
    # Determine the expected filename pattern based on last reconstruction
    expected_filename = [f"{st.session_state.last_dataset}_{st.session_state.last_topology}.ply", 
                         f"{st.session_state.last_dataset}"]
    st.info(f"Looking for results from last reconstruction: {expected_filename}")
    
    # Search for matching PLY files in all output directories
    matching_ply_files = []
    for dir_name in output_dirs:
        dir_path = OUTPUT_DIR / dir_name
        for ply_file in dir_path.glob("*.ply"):
            for expected_file in expected_filename:
                if expected_file in ply_file.name:
                    matching_ply_files.append((ply_file, os.path.getmtime(ply_file)))
    
    # # If no exact match found, try partial match
    # if not matching_ply_files:
    #     for dir_name in output_dirs:
    #         dir_path = OUTPUT_DIR / dir_name
    #         for ply_file in dir_path.glob("*.ply"):
    #             if st.session_state.last_dataset in ply_file.name and st.session_state.last_topology in ply_file.name:
    #                 matching_ply_files.append((ply_file, os.path.getmtime(ply_file)))
    
    # # If still no matches, try any file with the dataset name
    # if not matching_ply_files:
    #     for dir_name in output_dirs:
    #         dir_path = OUTPUT_DIR / dir_name
    #         for ply_file in dir_path.glob("*.ply"):
    #             if st.session_state.last_dataset in ply_file.name:
    #                 matching_ply_files.append((ply_file, os.path.getmtime(ply_file)))
    
        if matching_ply_files:
            # Sort by modification time (newest first)
            matching_ply_files.sort(key=lambda x: x[1], reverse=True)
            
            # Create a list of file options with descriptive names
            file_options = []
            for ply_file, mod_time in matching_ply_files:
                # Format the modification time
                mod_time_str = os.path.getmtime(ply_file)
                from datetime import datetime
                mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                
                # Create a descriptive name including file location
                location = "main output" if ply_file.parent == OUTPUT_DIR else str(ply_file.parent.relative_to(OUTPUT_DIR))
                desc_name = f"{ply_file.name} ({location}, modified {mod_time_str})"
                file_options.append((desc_name, ply_file))
            
            # Default to the newest file
            default_index = 0
            
            # Let user select which file to view
            selected_desc = st.selectbox(
                "Select Point Cloud File", 
                [desc for desc, _ in file_options],
                index=default_index
            )
            
            # Find the selected file
            selected_ply = next(ply_file for desc, ply_file in file_options if desc == selected_desc)
            
            # Set results directory to where this file was found
            results_dir = selected_ply.parent
            st.success(f"Showing results from {results_dir.relative_to(OUTPUT_DIR) if results_dir != OUTPUT_DIR else 'main output directory'}")
            
            # Display the point cloud
            st.subheader(f"Point Cloud: {selected_ply.name}")
            display_point_cloud(str(selected_ply))
    else:
        st.warning(f"No matching PLY files found for {expected_filename}. Please run a reconstruction first or switch to 'Browse All Files' mode.")
        # Default to main output directory
        results_dir = OUTPUT_DIR
    
    # Display disparity maps and other visualizations
    st.subheader("Visualizations")
    
    # Get all PNG files in the results directory
    image_files = list(results_dir.glob("*.png"))
    
    if not image_files:
        st.warning(f"No visualization images found in {results_dir}")
    else:
        # Group images by type
        disparity_maps = [str(f) for f in image_files if "disparity" in str(f).lower()]
        point_cloud_views = [str(f) for f in image_files if "pointcloud" in str(f).lower()]
        rectification = [str(f) for f in image_files if "rectification" in str(f).lower()]
        other_images = [str(f) for f in image_files if 
                       f not in disparity_maps and 
                       f not in point_cloud_views and 
                       f not in rectification]
        
        # Create expandable sections for each visualization type
        with st.expander("Disparity Maps", expanded=True):
            if disparity_maps:
                display_image_grid(disparity_maps, cols=2)
            else:
                st.info("No disparity maps found")
                
        with st.expander("Point Cloud Views", expanded=True):
            if point_cloud_views:
                display_image_grid(point_cloud_views, cols=2)
            else:
                st.info("No point cloud views found")
                
        with st.expander("Rectification", expanded=False):
            if rectification:
                display_image_grid(rectification, cols=2)
            else:
                st.info("No rectification visualizations found")
                
        # with st.expander("Other Visualizations", expanded=False):
        #     if other_images:
        #         display_image_grid(other_images, cols=2)
        #     else:
        #         st.info("No additional visualizations found")

# ABOUT PAGE
elif page == "About":
    st.header("About This Project")
    
    st.markdown("""
    ## 3D Reconstruction with Uncalibrated Stereo
    
    ### Problem Statement
    As a drone delivery company, we face challenges in efficiently navigating urban environments with complex structures and obstacles. Traditional 2D mapping techniques may not provide sufficient situational awareness, hindering safe and precise delivery operations. To address this, we need a robust 3D reconstruction solution that can accurately model the 3D environment, enabling our drones to plan optimal flight paths, detect obstacles, and ensure reliable and timely deliveries in densely populated areas.
    
    ### Abstract
    In our study, we evaluated the accuracy and efficiency of Multi-View Stereo (MVS) and Structure-from-Motion (SFM) technologies for 3D reconstruction. We utilized publicly available online datasets to assess the performance of these methods in generating detailed 3D models. The results highlighted MVS's proficiency in producing dense and intricate reconstructions. Meanwhile, SFM demonstrated greater efficiency in scenarios where a wide range of viewpoints was involved.
    
    ### Implementation
    To implement 3D reconstruction on the drone, we integrate cameras under the drone to capture multiple images of the environment from different viewpoints as the drone flies. These images are processed using:
    
    1. **Stereo Rectification** - Aligns image pairs to simplify the correspondence problem
    2. **Disparity Map Calculation** - Computes pixel-wise disparities between stereo image pairs
    3. **3D Reprojection** - Converts 2D disparities to 3D points
    4. **Point Cloud Alignment** - Uses ICP (Iterative Closest Point) to merge point clouds
    5. **Surface Reconstruction** - Creates a 3D mesh from aligned point clouds
    
    ### Resources
    - [GitHub Repository](https://github.com/yourusername/3D-Reconstruction-with-Uncalibrated-Stereo)
    - [Paper: Multi-View Stereo: A Tutorial](https://www.nowpublishers.com/article/Details/CGV-023)
    """)
    
    # Project contributors
    st.subheader("Contributors")
    st.markdown("""
    - Your Name
    - Team Member 2
    - Team Member 3
    """)
