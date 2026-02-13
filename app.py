import streamlit as st
import pandas as pd
import pydeck as pdk
import networkx as nx
import osmnx as ox
import numpy as np
from geopy.geocoders import ArcGIS
from streamlit_js_eval import get_geolocation
from streamlit_autorefresh import st_autorefresh
from geopy.distance import geodesic
import zipfile

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Safe Walk App | Corporate Edition")

# --- 1. MEMORY MANAGEMENT ---
if 'calculated_routes' not in st.session_state:
    st.session_state.calculated_routes = None
if 'origin_coords' not in st.session_state:
    st.session_state.origin_coords = None
if 'dest_coords' not in st.session_state:
    st.session_state.dest_coords = None
if 'last_gps_coords' not in st.session_state:
    st.session_state.last_gps_coords = None

# --- 2. DATA LOADING ---
@st.cache_resource
def load_graph():
    try:
        G = ox.load_graphml('nyc_advanced.graphml')
    except FileNotFoundError:
        try:
            G = ox.load_graphml('manhattan_advanced.graphml')
        except:
            st.error("System Error: Map data not found.")
            st.stop()
    
    for u, v, data in G.edges(data=True):
        if 'length' in data:
            try:
                val = data['length']
                data['length'] = float(val[0]) if isinstance(val, list) else float(val)
            except: data['length'] = 10.0
        for attr in ['crime_w', 'safe_w', 'street_type']:
            if attr in data:
                try: data[attr] = float(data[attr])
                except: data[attr] = 0.0
    return G

@st.cache_data
def load_crime_data():
    try:
        df = pd.read_csv('processed_risk_data.csv')
        df['Latitude'] = pd.to_numeric(df['Latitude'].astype(str).str.replace(',', '.'), errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'].astype(str).str.replace(',', '.'), errors='coerce')
        df = df.dropna(subset=['Latitude', 'Longitude'])
        ruido_lat = np.random.normal(0, 0.00015, size=len(df))
        ruido_lon = np.random.normal(0, 0.00015, size=len(df))
        df['lat_visual'] = df['Latitude'] + ruido_lat
        df['lon_visual'] = df['Longitude'] + ruido_lon
        return df
    except: return pd.DataFrame() 

@st.cache_data
def load_safe_places():
    try:
        df = pd.read_csv('safe_places.csv')
        df['name'] = df['name'].fillna('Safe Haven')
        return df
    except: return pd.DataFrame()

try:
    G = load_graph()
    crime_df = load_crime_data()
    safe_places_df = load_safe_places()
    geolocator = ArcGIS() 
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# --- HELPER FUNCTIONS ---
def calculate_metrics(G, route_nodes):
    dist_m = nx.path_weight(G, route_nodes, weight='length')
    time_min = int(dist_m / 83.0) 
    return dist_m, time_min

def get_route_geometry(G, route_nodes):
    coords = []
    for node in route_nodes:
        point = G.nodes[node]
        coords.append([point['x'], point['y']])
    return coords

# NUEVA FUNCIÓN PARA FORMATEAR EL TIEMPO
def format_time(minutes):
    if minutes >= 60:
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours} h {mins} min"
    else:
        return f"{minutes} min"

def get_turn_by_turn(G, route_nodes):
    instructions = []
    if not route_nodes or len(route_nodes) < 2:
        return ["Walk to destination"]
    
    current_street = None
    segment_dist = 0.0
    
    for i in range(len(route_nodes) - 1):
        u = route_nodes[i]
        v = route_nodes[i+1]
        
        edge_data = G.get_edge_data(u, v)
        if edge_data:
            data = edge_data.get(0, edge_data)
            street_name = data.get('name', 'Unnamed Path')
            if isinstance(street_name, list): 
                street_name = street_name[0]
            
            length = data.get('length', 10.0)
            
            if street_name == current_street:
                segment_dist += length
            else:
                if current_street:
                    # Quitamos negritas y cursivas para que quede limpio en el DataFrame
                    instructions.append(f"Travel {int(segment_dist)}m on {current_street}")
                current_street = street_name
                segment_dist = length
    
    if current_street:
        instructions.append(f"Travel {int(segment_dist)}m on {current_street}")
        
    instructions.append("Arrive at Destination")
    return instructions

# --- SIDEBAR UI ---
st.sidebar.title("Route Planner")

# --- 🚨 SOS BUTTON (EMERGENCY) ---
st.sidebar.markdown("### Emergency")
sos_container = st.sidebar.container()
if sos_container.button("🚨 SOS / PANIC", type="primary", use_container_width=True):
    lat_sos = st.session_state.last_gps_coords[0] if st.session_state.last_gps_coords else "Unknown"
    lon_sos = st.session_state.last_gps_coords[1] if st.session_state.last_gps_coords else "Unknown"
    st.toast("EMERGENCY MODE ACTIVATED", icon="🚨")
    st.sidebar.error(f"""
    **EMERGENCY ASSISTANCE**
    📍 **YOUR LOCATION:**
    `{lat_sos}, {lon_sos}`
    📞 **CALLING 911...**
    """)
    st.markdown('<meta http-equiv="refresh" content="0; url=tel:911">', unsafe_allow_html=True)

st.sidebar.markdown("---") 

# --- GPS MODULE ---
st.sidebar.markdown("### Location Services")
use_live_gps = st.sidebar.checkbox("Enable Live GPS Tracking")

if use_live_gps:
    st_autorefresh(interval=3000, key="gps_refresher")

gps_data = get_geolocation() 
default_origin = "Times Square, Manhattan, NY" 
lat_gps, lon_gps = None, None
gps_valid = False

if gps_data and 'coords' in gps_data:
    lat_gps = gps_data['coords']['latitude']
    lon_gps = gps_data['coords']['longitude']
    st.session_state.last_gps_coords = (lat_gps, lon_gps)
elif st.session_state.last_gps_coords:
    lat_gps, lon_gps = st.session_state.last_gps_coords

if lat_gps and lon_gps:
    nyc_center = (40.7580, -73.9855)
    dist_to_nyc = geodesic((lat_gps, lon_gps), nyc_center).km
    
    if dist_to_nyc > 50:
        st.sidebar.warning(f"⚠️ GPS detected ({dist_to_nyc:.0f}km away). Too far for local routing.")
        gps_valid = False
    else:
        st.sidebar.info(f"📍 GPS Active within NYC")
        gps_valid = True
        if use_live_gps:
            default_origin = "Current GPS Location"
        elif 'last_address_coords' not in st.session_state or st.session_state.last_address_coords != (lat_gps, lon_gps):
            try:
                location_info = geolocator.reverse((lat_gps, lon_gps), timeout=2)
                if location_info:
                    default_origin = location_info.address
                    st.session_state.last_address_coords = (lat_gps, lon_gps)
            except: pass

# --- INPUTS ---
origin_str = st.sidebar.text_input("Origin", value=default_origin)
dest_str = st.sidebar.text_input("Destination", "One World Trade Center, Manhattan, NY")

st.sidebar.markdown("###") 

btn_calculate = st.sidebar.button("Calculate Route", type="primary") 

st.sidebar.markdown("---")

# --- PREFERENCES ---
st.sidebar.markdown("### Algorithm Preferences")
check_crime = st.sidebar.checkbox("Avoid High-Risk Zones", value=True)
check_safe = st.sidebar.checkbox("Proximity to Safe Havens (24h)", value=False)
check_avenues = st.sidebar.checkbox("Prioritize Main Avenues", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Map Layers")

show_heatmap = st.sidebar.checkbox("Show Risk Heatmap", value=False)
usar_3d = st.sidebar.checkbox("3D Perspective", value=False)

st.sidebar.markdown("###") 
dark_mode = st.sidebar.toggle("Dark Mode", value=False) 

def dynamic_weight(u, v, d):
    cost = d.get('length', 10.0)
    if check_avenues: cost *= d.get('street_type', 1.0)
    if check_crime: cost += (d.get('crime_w', 0.0) * 10.0)
    if check_safe: cost -= (d.get('safe_w', 0.0) * 20.0) 
    return max(cost, 1.0)

# LOGICA DE CALCULO
auto_calc = use_live_gps and gps_valid and dest_str

if btn_calculate or auto_calc:
    with st.empty():
        if not use_live_gps: st.caption("Processing route logic...")
        
        try:
            start_coord = None
            if gps_valid and (origin_str == "Current GPS Location" or origin_str == default_origin) and lat_gps:
                start_coord = [lon_gps, lat_gps]
            else:
                search_query = origin_str
                if "NY" not in search_query and "New York" not in search_query:
                    search_query += ", New York City, NY"
                loc_origin = geolocator.geocode(search_query, timeout=5)
                if loc_origin: start_coord = [loc_origin.longitude, loc_origin.latitude]
                else: st.error(f"Could not locate origin: {origin_str}")

            search_dest = dest_str
            if "NY" not in search_dest and "New York" not in search_dest:
                 search_dest += ", New York City, NY"
            loc_dest = geolocator.geocode(search_dest, timeout=5)
            
            if start_coord and loc_dest:
                end_coord = [loc_dest.longitude, loc_dest.latitude]
                st.session_state.origin_coords = start_coord
                st.session_state.dest_coords = end_coord

                orig_node = ox.distance.nearest_nodes(G, start_coord[0], start_coord[1])
                dest_node = ox.distance.nearest_nodes(G, end_coord[0], end_coord[1])

                route_custom = nx.shortest_path(G, orig_node, dest_node, weight=dynamic_weight)
                dist_custom, time_custom = calculate_metrics(G, route_custom)
                geom_custom = get_route_geometry(G, route_custom)
                
                route_fast = nx.shortest_path(G, orig_node, dest_node, weight='length')
                dist_fast, time_fast = calculate_metrics(G, route_fast)
                geom_fast = get_route_geometry(G, route_fast)
                
                steps_custom = get_turn_by_turn(G, route_custom)
                steps_fast = get_turn_by_turn(G, route_fast)

                st.session_state.calculated_routes = {
                    "custom": {
                        "geom": geom_custom, "dist": dist_custom, "time": time_custom, 
                        "steps": steps_custom, "nodes": route_custom
                    },
                    "fast": {
                        "geom": geom_fast, "dist": dist_fast, "time": time_fast,
                        "steps": steps_fast, "nodes": route_fast
                    }
                }
        except Exception as e:
            if not use_live_gps: st.error(f"Routing Error: {e}")

# --- MAIN INTERFACE: BRANDING TITLE ---
st.markdown("""
<style>
.main-title {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-weight: 800;
    font-size: 3.5rem;
    background: linear-gradient(90deg, #2E3192 0%, #1BFFFF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
}
.subtitle {
    font-size: 1.2rem;
    color: #666;
    margin-bottom: 30px;
    font-weight: 300;
}
</style>
<h1 class="main-title">Safe Walk App</h1>
<div class="subtitle">Advanced Urban Navigation System</div>
""", unsafe_allow_html=True)


layers = []
hm_opacity = 0.25 if dark_mode else 0.6 

if show_heatmap and not crime_df.empty:
    layers.append(pdk.Layer("HeatmapLayer", data=crime_df, get_position='[lon_visual, lat_visual]', opacity=hm_opacity, radius_pixels=40, intensity=1, threshold=0.05))

if check_safe and not safe_places_df.empty:
    layers.append(pdk.Layer("ScatterplotLayer", data=safe_places_df, get_position='[lon, lat]', get_color=[34, 139, 34, 180], get_radius=20, pickable=True, auto_highlight=True))

if st.session_state.calculated_routes:
    data = st.session_state.calculated_routes
    if use_live_gps and gps_valid:
        st.info(f"NAVIGATION ACTIVE: {format_time(data['custom']['time'])} remaining")
    else:
        # --- NUEVO DISEÑO DE BLOQUES SEPARADOS ---
        col_safe, col_fast = st.columns(2)

        # Bloque Safe Route (Izquierda)
        with col_safe:
            st.success("🛡️ **Safe Route (Recommended)**")
            st.metric("Estimated Time", format_time(data['custom']['time']), f"{data['custom']['dist']/1000:.2f} km")
            with st.expander("📄 Turn-by-Turn Directions"):
                # Usamos un DataFrame para tener una tabla limpia con scroll
                df_steps_safe = pd.DataFrame(data['custom']['steps'], columns=["Instruction"])
                st.dataframe(df_steps_safe, hide_index=True, use_container_width=True)
        
        # Bloque Fastest Route (Derecha)
        with col_fast:
            st.info("**Fastest Route**")
            st.metric("Estimated Time", format_time(data['fast']['time']), f"{data['fast']['dist']/1000:.2f} km")
            with st.expander("📄 Turn-by-Turn Directions"):
                # Usamos un DataFrame para tener una tabla limpia con scroll
                df_steps_fast = pd.DataFrame(data['fast']['steps'], columns=["Instruction"])
                st.dataframe(df_steps_fast, hide_index=True, use_container_width=True)
        
        st.markdown("---") # Separador antes de los controles del mapa

    
    view_mode = st.radio("Display Mode:", ["Compare Routes", "Safe Route Only", "Fastest Route Only"], horizontal=True)

    if view_mode in ["Compare Routes", "Safe Route Only"]:
        layers.append(pdk.Layer("PathLayer", data=[{"path": data['custom']['geom'], "name": "Safe Route"}], get_path="path", get_color=[46, 204, 113], width_scale=20, width_min_pixels=4, pickable=True))

    if view_mode in ["Compare Routes", "Fastest Route Only"]:
        layers.append(pdk.Layer("PathLayer", data=[{"path": data['fast']['geom'], "name": "Standard Route"}], get_path="path", get_color=[52, 152, 219], width_scale=20, width_min_pixels=4, pickable=True))

    origin_color = [0, 102, 204] if (use_live_gps and gps_valid) else [46, 204, 113]
    points_data = [
        {"pos": st.session_state.origin_coords, "color": [255, 255, 255], "rad": 40, "name": "Origin"},
        {"pos": st.session_state.origin_coords, "color": origin_color, "rad": 20, "name": "Origin"},
        {"pos": st.session_state.dest_coords, "color": [255, 255, 255], "rad": 40, "name": "Destination"},
        {"pos": st.session_state.dest_coords, "color": [220, 53, 69], "rad": 20, "name": "Destination"}
    ]
    layers.append(pdk.Layer("ScatterplotLayer", data=points_data, get_position="pos", get_color="color", get_radius="rad", pickable=True, opacity=1))

pitch_val = 45 if usar_3d else 0
map_style_val = pdk.map_styles.CARTO_DARK if dark_mode else pdk.map_styles.CARTO_LIGHT

if st.session_state.origin_coords:
    view_state = pdk.ViewState(latitude=st.session_state.origin_coords[1], longitude=st.session_state.origin_coords[0], zoom=14, pitch=pitch_val)
else:
    view_state = pdk.ViewState(latitude=40.73, longitude=-73.93, zoom=11, pitch=pitch_val)

st.pydeck_chart(pdk.Deck(
    map_style=map_style_val,
    initial_view_state=view_state,
    layers=layers,
    height=750,
    tooltip={"html": "<b>{name}</b>", "style": {"backgroundColor": "white", "color": "black", "font-family": "Helvetica Neue, Arial", "z-index": "1000"}}
))

# --- FOOTER CORPORATIVO ---
footer_bg = "#1E1E1E" if dark_mode else "#F0F2F6"
footer_text = "#FFFFFF" if dark_mode else "#333333"

st.markdown(f"""
<style>
.footer-banner {{
    background-color: {footer_bg};
    padding: 40px;
    border-radius: 12px;
    margin-top: 60px;
    display: flex;
    flex-direction: column;
    align-items: center;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    color: {footer_text};
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    transition: background-color 0.3s ease;
}}
.partner-title {{
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 35px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    opacity: 0.9;
}}
.logo-container {{
    display: flex;
    justify_content: center;
    gap: 80px;
    flex-wrap: wrap;
    align-items: end;
}}
.logo-item {{
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 140px;
}}
.logo-link img {{
    height: 80px;
    width: auto;
    margin-bottom: 15px;
    transition: transform 0.3s ease, opacity 0.3s ease;
    opacity: 0.85;
    filter: brightness(0.95); 
}}
.logo-link:hover img {{
    transform: scale(1.15);
    opacity: 1;
    filter: brightness(1);
}}
.logo-caption {{
    font-size: 14px;
    font-weight: 500;
    opacity: 0.7;
    margin-top: 5px;
}}
</style>

<div class="footer-banner">
    <div class="partner-title">Official Data Partners & Collaborations</div>
    <div class="logo-container">
        <div class="logo-item">
            <a href="https://opendata.cityofnewyork.us/" target="_blank" class="logo-link">
                <img src="https://opendata.cityofnewyork.us/wp-content/themes/opendata-wp/assets/img/nyc-open-data-logo.svg" alt="NYC Seal">
            </a>
            <div class="logo-caption">NYC OpenData</div>
        </div>
        <div class="logo-item">
                <a href="https://www.esade.edu/" target="_blank" class="logo-link">
                <img src="data:image/jpeg;base64,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" alt="ESADE Logo" style="border-radius: 8px;">
            </a>
            <div class="logo-caption">Academic Partner</div>
        </div>
        <div class="logo-item">
                <a href="https://www.openstreetmap.org/" target="_blank" class="logo-link">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Openstreetmap_logo.svg/320px-Openstreetmap_logo.svg.png" alt="OSM Logo">
            </a>
            <div class="logo-caption">Geospatial Data</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)