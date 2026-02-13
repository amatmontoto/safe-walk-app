import streamlit as st
import pandas as pd
import pydeck as pdk
import networkx as nx
import osmnx as ox
import numpy as np
from geopy.geocoders import ArcGIS 

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(layout="wide", page_title="NYC SafeTrek - Full City")

# --- 1. GESTIÓN DE MEMORIA ---
if 'rutas_calculadas' not in st.session_state:
    st.session_state.rutas_calculadas = None
if 'coords_origen' not in st.session_state:
    st.session_state.coords_origen = None
if 'coords_destino' not in st.session_state:
    st.session_state.coords_destino = None

# --- 2. CARGA DE DATOS ---
@st.cache_resource
def load_graph():
    try:
        # CAMBIO IMPORTANTE: Cargamos el mapa de TODA la ciudad
        G = ox.load_graphml('nyc_advanced.graphml')
    except FileNotFoundError:
        st.error("⚠️ No encuentro 'nyc_advanced.graphml'. Asegúrate de que el script 3 terminó bien.")
        st.stop()
        
    for u, v, data in G.edges(data=True):
        if 'length' in data:
            try:
                val = data['length']
                data['length'] = float(val[0]) if isinstance(val, list) else float(val)
            except: data['length'] = 10.0
            
        for attr in ['crime_w', 'safe_w', 'street_type']:
            if attr in data:
                try:
                    data[attr] = float(data[attr])
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
    except:
        return pd.DataFrame() 

@st.cache_data
def load_safe_places():
    try:
        df = pd.read_csv('safe_places.csv')
        df['name'] = df['name'].fillna('Refugio Seguro')
        return df
    except:
        return pd.DataFrame()

try:
    G = load_graph()
    crime_df = load_crime_data()
    safe_places_df = load_safe_places()
    geolocator = ArcGIS() 
except Exception as e:
    st.error(f"Error iniciando: {e}")
    st.stop()

# --- FUNCIONES AUXILIARES ---
def calcular_metricas(G, route_nodes):
    dist_m = nx.path_weight(G, route_nodes, weight='length')
    tiempo_min = int(dist_m / 83.0) 
    return dist_m, tiempo_min

def get_route_geometry(G, route_nodes):
    coords = []
    for node in route_nodes:
        point = G.nodes[node]
        coords.append([point['x'], point['y']])
    return coords

# --- INTERFAZ ---
st.sidebar.header("📍 Planificador de Ruta (NYC Completo)")
origin_str = st.sidebar.text_input("Origen", "Times Square, New York")
dest_str = st.sidebar.text_input("Destino", "Barclays Center, Brooklyn") # Ejemplo inter-condados

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Preferencias de Seguridad")

check_crime = st.sidebar.checkbox("Evitar Zonas de Riesgo", value=True, help="Evita zonas con historial de crímenes violentos.")
check_safe = st.sidebar.checkbox("Buscar Refugios (24h)", value=False, help="Pasa cerca de hospitales, comisarías y tiendas abiertas.")
check_avenues = st.sidebar.checkbox("Priorizar Avenidas", value=True, help="Prefiere calles anchas e iluminadas frente a callejones.")

def dynamic_weight(u, v, d):
    cost = d.get('length', 10.0)
    
    if check_avenues:
        factor = d.get('street_type', 1.0)
        cost *= factor
    
    if check_crime:
        crime = d.get('crime_w', 0.0)
        cost += (crime * 10.0)
    
    if check_safe:
        safe = d.get('safe_w', 0.0)
        cost -= (safe * 20.0) 
    
    return max(cost, 1.0)

if st.sidebar.button("🔎 Buscar Rutas"):
    with st.spinner("Calculando ruta cruzando la ciudad..."):
        try:
            loc_origin = geolocator.geocode(origin_str, timeout=10)
            loc_dest = geolocator.geocode(dest_str, timeout=10)

            if loc_origin and loc_dest:
                start_coord = [loc_origin.longitude, loc_origin.latitude]
                end_coord = [loc_dest.longitude, loc_dest.latitude]
                
                st.session_state.coords_origen = start_coord
                st.session_state.coords_destino = end_coord

                orig_node = ox.distance.nearest_nodes(G, start_coord[0], start_coord[1])
                dest_node = ox.distance.nearest_nodes(G, end_coord[0], end_coord[1])

                route_custom = nx.shortest_path(G, orig_node, dest_node, weight=dynamic_weight)
                dist_custom, time_custom = calcular_metricas(G, route_custom)
                geom_custom = get_route_geometry(G, route_custom)

                route_fast = nx.shortest_path(G, orig_node, dest_node, weight='length')
                dist_fast, time_fast = calcular_metricas(G, route_fast)
                geom_fast = get_route_geometry(G, route_fast)

                st.session_state.rutas_calculadas = {
                    "custom": {"geom": geom_custom, "dist": dist_custom, "time": time_custom},
                    "fast": {"geom": geom_fast, "dist": dist_fast, "time": time_fast}
                }
            else:
                st.error("No se encontraron las direcciones.")
        except Exception as e:
            st.error(f"Error calculando: {e}")

# --- PANTALLA PRINCIPAL ---
st.title("🗽 NYC SafeTrek")

layers = []

# 1. HEATMAP
if not crime_df.empty:
    layers.append(pdk.Layer(
        "HeatmapLayer",
        data=crime_df,
        get_position='[lon_visual, lat_visual]', 
        opacity=0.3,
        radius_pixels=30,
        intensity=1,
        threshold=0.05
    ))

# 2. REFUGIOS
if check_safe and not safe_places_df.empty:
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=safe_places_df,
        get_position='[lon, lat]',
        get_color=[0, 255, 0, 160], 
        get_radius=15,             
        pickable=True
    ))

# 3. RUTAS
if st.session_state.rutas_calculadas:
    data = st.session_state.rutas_calculadas
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🛡️ **Ruta Segura**\n\n⏱️ {data['custom']['time']} min | 📏 {data['custom']['dist']/1000:.2f} km")
    with col2:
        st.info(f"⚡ **Ruta Rápida**\n\n⏱️ {data['fast']['time']} min | 📏 {data['fast']['dist']/1000:.2f} km")

    st.write("---")
    modo = st.radio("Modo Visualización:", ["Comparar Ambas", "Solo Segura", "Solo Rápida"], horizontal=True)

    if modo in ["Comparar Ambas", "Solo Segura"]:
        layers.append(pdk.Layer(
            "PathLayer",
            data=[{"path": data['custom']['geom']}],
            get_path="path",
            get_color=[0, 255, 128], 
            width_scale=20,
            width_min_pixels=3,
            pickable=True
        ))

    if modo in ["Comparar Ambas", "Solo Rápida"]:
        layers.append(pdk.Layer(
            "PathLayer",
            data=[{"path": data['fast']['geom']}],
            get_path="path",
            get_color=[0, 100, 255], 
            width_scale=20,
            width_min_pixels=3,
            pickable=True
        ))

    points_data = [
        {"pos": st.session_state.coords_origen, "color": [255, 255, 255], "rad": 30},
        {"pos": st.session_state.coords_origen, "color": [0, 255, 0], "rad": 15},
        {"pos": st.session_state.coords_destino, "color": [255, 255, 255], "rad": 30},
        {"pos": st.session_state.coords_destino, "color": [255, 0, 0], "rad": 15}
    ]
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=points_data,
        get_position="pos",
        get_color="color",
        get_radius="rad",
        pickable=True,
        opacity=1
    ))

usar_3d = st.sidebar.checkbox("Vista 3D", value=False)
pitch_val = 45 if usar_3d else 0

if st.session_state.coords_origen:
    view_state = pdk.ViewState(
        latitude=st.session_state.coords_origen[1],
        longitude=st.session_state.coords_origen[0],
        zoom=13.5,
        pitch=pitch_val
    )
else:
    # Vista por defecto centrada para ver toda la ciudad
    view_state = pdk.ViewState(latitude=40.73, longitude=-73.93, zoom=11, pitch=pitch_val)

st.pydeck_chart(pdk.Deck(
    map_style=pdk.map_styles.CARTO_DARK,
    initial_view_state=view_state,
    layers=layers,
    tooltip={
        "html": "<b>{name}</b>",
        "style": {"color": "white"}
    }
))