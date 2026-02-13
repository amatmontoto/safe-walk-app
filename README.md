# Safe Walk App - NYC Intelligent Navigation

Safe Walk App is an advanced urban navigation prototype designed to route pedestrians through New York City prioritizing safety over speed.

Using data from **NYC OpenData** (Crime Complaints) and **OpenStreetMap**, the application uses a dynamic Dijkstra algorithm to weigh street safety in real-time based on historical crime data, lighting conditions (estimated by street type), and proximity to safe havens (hospitals, police stations).

## Features
* **Dual Routing:** Compares "Fastest Route" vs. "Safe Route".
* **Live GPS:** Real-time location tracking for navigation.
* **Emergency Mode:** SOS button with coordinate broadcasting.
* **Turn-by-Turn Directions:** Detailed textual instructions.
* **3D Data Visualization:** Interactive risk heatmaps.

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/TU_USUARIO/safe-walk-app.git](https://github.com/TU_USUARIO/safe-walk-app.git)
    cd safe-walk-app
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

*Note: The application will automatically unzip the high-resolution NYC graph map (`nyc_advanced.zip`) on the first run.*

## Project Structure
* `app.py`: Main application runtime (Streamlit).
* `nyc_advanced.zip`: Compressed graph data of NYC (Nodes/Edges with risk weights).
* `processed_risk_data.csv`: Cleaned dataset of crime incidents.
* `safe_places.csv`: Geolocation of hospitals, police stations, and 24/7 shelters.
* `3_build_advanced_graph.py`: (Offline) Script used to generate the graph and train the risk weights.
