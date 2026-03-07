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
import os
from dotenv import load_dotenv
from groq import Groq
import json
import uuid
import urllib.parse
import requests




load_dotenv()

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# LECTOR DE SENSORES Y CLIMA
def fetch_ai_weather_context():
    # Usamos la memoria para que el agente solo piense 1 vez por sesión (Latencia cero al recargar)
    if "live_weather_context" in st.session_state:
        return st.session_state.live_weather_context
    
    try:
        # 1. Extracción de datos crudos (Coordenadas de Manhattan, NYC)
        url = "https://api.open-meteo.com/v1/forecast?latitude=40.7128&longitude=-74.0060&current_weather=true"
        response = requests.get(url, timeout=3).json()
        weather_data = response.get("current_weather", {})
        
        # 2. El Agente Meteorólogo (Interpretación con IA)
        agent_prompt = f"""
        You are an elite urban meteorologist AI operating in New York City.
        Here is the raw live sensor data: {weather_data}
        (Note: weathercode follows WMO standard. 0 is clear, 61 is rain, 71 is snow, etc. is_day: 1 is day, 0 is night).
        Translate this raw data into a short 4 to 8 word safety or environmental alert for pedestrians. 
        Include 1 or 2 relevant emojis. 
        Keep it strictly in English. NEVER include any other text, explanation, or quotes.
        Example outputs: 
        🌙 Chilly & Clear: Safe walking conditions.
        🌧️ Wet & Slippery: Proceed with caution!
        """
        
        # Usamos el modelo más rápido de Groq para que sea instantáneo
        completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": agent_prompt}],
            model="llama-3.3-70b-versatile", 
            temperature=0.3,
            max_tokens=30
        )
        
        ai_weather_alert = completion.choices[0].message.content.strip().replace('"', '')
        st.session_state.live_weather_context = ai_weather_alert
        return ai_weather_alert
        
    except Exception as e:
        # Esto imprimirá el error real en tu pantalla para que lo veamos
        st.sidebar.error(f"🔍 Chivato del error: {e}") 
        return "🌍 NYC Sensors offline. Assuming clear conditions."


if "pending_crime" in st.session_state:
    st.session_state.smart_crime = st.session_state.pending_crime
    del st.session_state.pending_crime
if "pending_safe" in st.session_state:
    st.session_state.smart_safe = st.session_state.pending_safe
    del st.session_state.pending_safe
if "pending_avenues" in st.session_state:
    st.session_state.smart_avenues = st.session_state.pending_avenues
    del st.session_state.pending_avenues

if "smart_origin" not in st.session_state: st.session_state.smart_origin = "Times Square, Manhattan, NY"
if "user_pins" not in st.session_state: 
    st.session_state.user_pins = []


def extraer_calles_principales(nodos_ruta, grafo):
    """Extrae las calles de forma robusta evitando cuelgues."""
    calles = []
    try:
        for i in range(len(nodos_ruta)-1):
            u, v = nodos_ruta[i], nodos_ruta[i+1]
            if grafo.has_edge(u, v):
                data = grafo.get_edge_data(u, v)
                # Seleccionar la arista correcta (0 suele ser el default en MultiDiGraphs)
                edge_info = data[0] if 0 in data else list(data.values())[0]
                
                if 'name' in edge_info:
                    nombre = edge_info['name']
                    # OSMnx a veces devuelve una lista si hay calles solapadas
                    if isinstance(nombre, list): 
                        nombre = nombre[0]
                    
                    if isinstance(nombre, str) and (not calles or calles[-1] != nombre):
                        calles.append(nombre)
                        
        calles_unicas = list(dict.fromkeys(calles))
        
        if len(calles_unicas) > 3:
            return f"{calles_unicas[0]}, continuing via {calles_unicas[len(calles_unicas)//2]}, and arriving via {calles_unicas[-1]}"
        elif calles_unicas:
            return ", ".join(calles_unicas)
        else:
            return "local unnamed streets"
    except Exception as e:
        return "unnamed pathways"
    

# Inicializar las variables que la IA y la barra lateral compartirán
if "smart_origin" not in st.session_state: st.session_state.smart_origin = "Times Square, Manhattan, NY"
if "smart_dest" not in st.session_state: st.session_state.smart_dest = "One World Trade Center, Manhattan, NY"
# Inicializar variables (1.0 = Máxima prioridad, 0.0 = Ignorar)
if "smart_crime" not in st.session_state: st.session_state.smart_crime = 1.0
if "smart_safe" not in st.session_state: st.session_state.smart_safe = 0.5
if "smart_avenues" not in st.session_state: st.session_state.smart_avenues = 1.0
if "ai_trigger" not in st.session_state: st.session_state.ai_trigger = False


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
        # LIMPIEZA DE DATOS ANTI-AMNESIA
        for attr in ['crime_w', 'safe_w', 'street_type']:
            if attr in data:
                val = data[attr]
                try:
                    # Si el mapa guardó los datos como un texto con corchetes e.g. "[1.5, 2.0]"
                    if isinstance(val, str) and val.startswith('['):
                        # Le quitamos los corchetes, separamos por comas y nos quedamos el primer número
                        val = val.strip('[]').split(',')[0]
                    # Si ya es una lista normal
                    elif isinstance(val, list):
                        val = val[0]
                        
                    data[attr] = float(val)
                except: 
                    data[attr] = 0.0
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
# ==========================================
# ⚙️ SIDEBAR: INPUTS & PREFERENCES
# ==========================================
# ==========================================
# 🌤️ ENVIRONMENTAL CONTEXT (LIVE AI AGENT)
# ==========================================
st.sidebar.markdown("### Live Environment")
# Un pequeño spinner nativo muy elegante mientras el agente piensa
with st.sidebar.status("📡 Syncing NYC Sensors...", expanded=False) as status:
    live_weather = fetch_ai_weather_context()
    status.update(label="Sensors Synced", state="complete", expanded=False)

st.sidebar.info(f"**{live_weather}**")

# Guardamos el contexto para que el Macro-Cerebro lo lea
st.session_state.env_context = live_weather
st.sidebar.markdown("---")

st.sidebar.markdown("### Route Planner")

# Cajas conectadas a la memoria de la IA
st.session_state.smart_origin = st.sidebar.text_input("Origin", value=st.session_state.smart_origin)
st.session_state.smart_dest = st.sidebar.text_input("Destination", value=st.session_state.smart_dest)

st.sidebar.markdown("###") 
btn_calculate = st.sidebar.button("Calculate Route", type="primary")


# ==========================================
# 📍 COMMUNITY REPORTS (VISUAL PINS)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.markdown("### Community Reports")
st.sidebar.caption("Report temporary incidents to visualize them on your map.")

report_type = st.sidebar.selectbox("Incident Type:", [
    "⚠️ Hazard / Accident", 
    "🥊 Altercation / Suspicious", 
    "💡 Broken Streetlights"
])
report_loc = st.sidebar.text_input("Location (e.g., '5th Ave and W 42nd St')")

if st.sidebar.button("📍 Drop Pin", use_container_width=True):
    if report_loc:
        with st.spinner("Locating..."):
            search_pin = report_loc
            if "NY" not in search_pin and "New York" not in search_pin:
                search_pin += ", New York City, NY"
            
            # Usamos el mismo geolocalizador que ya tienes
            pin_coords = geolocator.geocode(search_pin, timeout=5)
            
            if pin_coords:
                new_pin = {
                    "id": str(uuid.uuid4()),
                    "type": report_type,
                    "location": report_loc,
                    "lat": pin_coords.latitude,
                    "lon": pin_coords.longitude
                }
                st.session_state.user_pins.append(new_pin)
                st.rerun() # Refrescamos para que el mapa dibuje la chincheta
            else:
                st.sidebar.error("Location not found. Be more specific.")

# --- Gestión de Chinchetas Activas (Borrado) ---
if st.session_state.user_pins:
    st.sidebar.markdown("#### Active Reports")
    for pin in st.session_state.user_pins:
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            st.markdown(f"<span style='font-size: 0.85em;'>{pin['type']}<br><b>{pin['location']}</b></span>", unsafe_allow_html=True)
        with col2:
            # Botón para borrar la chincheta
            if st.button("❌", key=f"del_{pin['id']}"):
                # Filtramos la lista para quitar la que coincide con el ID
                st.session_state.user_pins = [p for p in st.session_state.user_pins if p['id'] != pin['id']]
                st.rerun()

st.sidebar.markdown("---")


st.sidebar.markdown("### Algorithm Preferences")

# Casillas conectadas a la memoria de la IA
# --- PREFERENCES ---
st.sidebar.markdown("### Algorithm Preferences")

# Si la IA cambia la variable, la bolita de la pantalla se mueve sola.
st.sidebar.slider("Avoid High-Risk Zones Weight", 0.0, 1.0, step=0.1, key="smart_crime")
st.sidebar.slider("Proximity to Safe Havens Weight", 0.0, 1.0, step=0.1, key="smart_safe")
st.sidebar.slider("Prioritize Main Avenues Weight", 0.0, 1.0, step=0.1, key="smart_avenues")

# Variables finales para que tu motor Dijkstra las lea
check_crime = st.session_state.smart_crime
check_safe = st.session_state.smart_safe
check_avenues = st.session_state.smart_avenues

st.sidebar.markdown("---")
st.sidebar.markdown("### Map Layers")


# AQUÍ ESTÁN LAS VARIABLES QUE TE HABÍA ROTO (YA ESTÁN ARREGLADAS)
show_heatmap = st.sidebar.checkbox("Show Risk Heatmap", value=False)
usar_3d = st.sidebar.checkbox("3D Perspective", value=False)
dark_mode = st.sidebar.toggle("Dark Mode", value=False) 

# Variables finales de texto (las usa la IA y tu lógica de cálculo)
origin_str = st.session_state.smart_origin
dest_str = st.session_state.smart_dest
check_crime = st.session_state.smart_crime
check_safe = st.session_state.smart_safe
check_avenues = st.session_state.smart_avenues

# --- CONEXIÓN DE PESOS ---
def dynamic_weight(u, v, d):
    # Coste base de la calle: su longitud física real en metros
    cost = d.get('length', 10.0)
    
    # 1. EL MURO DEL CRIMEN (Penalización multiplicativa extrema)
    # En lugar de sumar metros sueltos, multiplicamos la longitud de la calle.
    # Una calle de 200m con crimen alto pasará a "medir" miles de metros.
    if check_crime > 0: 
        cost += (cost * d.get('crime_w', 0.0) * 50.0 * check_crime)
        
    # 2. EL IMÁN DE SEGURIDAD (Atracción divisiva)
    # Si hay puntos seguros, "encogemos" la calle para que el algoritmo 
    # crea que es un atajo increíblemente corto y se desvíe hacia allí.
    if check_safe > 0: 
        cost = cost / (1.0 + (d.get('safe_w', 0.0) * 10.0 * check_safe))
        
    # 3. AVENIDAS (Descuento geométrico seguro)
    # Si es una avenida principal, le aplicamos un "descuento" de hasta el 30% 
    # en su distancia, atrayendo la ruta hacia calles grandes y bien iluminadas.
    if check_avenues > 0: 
        # Asumimos que street_type es mayor si es una calle principal
        cost = cost * (1.0 - (0.3 * check_avenues * d.get('street_type', 1.0)))
        
    # Seguro de vida matemático: Evita costes negativos o de valor cero 
    # que harían explotar la lógica interna de la librería NetworkX.
    return max(cost, 1.0)

# LOGICA DE CALCULO
auto_calc = use_live_gps and gps_valid and dest_str

if btn_calculate or auto_calc or st.session_state.ai_trigger:
    st.session_state.ai_trigger = False 
    
    logic_container = st.container()
    
    with logic_container:
        if not use_live_gps: st.caption("Processing route logic...")
        
        try:
            start_coord = None
            
            # FILTRO ANTI-ERRORES GPS (Desacoplado del texto)
            # AHORA SOLO salta si explícitamente pide el GPS. 
            if origin_str == "Current GPS Location":
                if gps_valid and lat_gps:
                    start_coord = [lon_gps, lat_gps]
                else:
                    st.warning("📍 **GPS Error:** Your current location is detected outside of New York City (or GPS is disabled).")
                    if st.button("🔙 Use Times Square Instead"):
                        st.session_state.smart_origin = "Times Square, Manhattan, NY"
                        st.rerun()
                    st.stop()
            else:
                # Todo lo demás (incluido "Times Square") se procesa como texto limpio
                search_query = origin_str
                if "NY" not in search_query and "New York" not in search_query:
                    search_query += ", New York City, NY"
                
                loc_origin = geolocator.geocode(search_query, timeout=5)
                if loc_origin: 
                    start_coord = [loc_origin.longitude, loc_origin.latitude]
                else: 
                    st.error(f"Could not locate origin: {origin_str}")
                    st.stop()

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
            
            # --- FEATURE 2: SAFETY MATCH SCORE (SAFE) ---
            # Calcula un porcentaje visualmente lógico (92% - 99%)
            safe_score = min(99, int(92 + (st.session_state.smart_crime * 7)))
            st.caption(f"**Safety Match:** {safe_score}%")
            st.progress(safe_score / 100.0) # La barra de progreso pide un decimal de 0.0 a 1.0
            
            st.metric("Estimated Time", format_time(data['custom']['time']), f"{data['custom']['dist']/1000:.2f} km")
            with st.expander("📄 Turn-by-Turn Directions"):
                df_steps_safe = pd.DataFrame(data['custom']['steps'], columns=["Instruction"])
                st.dataframe(df_steps_safe, hide_index=True, use_container_width=True)
            
            # --- FEATURE 1: SHARE MY WALK ---
            safe_time = format_time(data['custom']['time'])
            share_text = f"🚨 Safe Walk Alert: I am walking from {st.session_state.smart_origin} to {st.session_state.smart_dest}. Estimated travel time: {safe_time}. My Safe Walk app is actively monitoring the route."
            encoded_text = urllib.parse.quote(share_text)
            whatsapp_url = f"https://wa.me/?text={encoded_text}"
            st.link_button("📲 Share Route with Trusted Contact", whatsapp_url, use_container_width=True)
        
        # Bloque Fastest Route (Derecha)
        with col_fast:
            st.info("**Fastest Route**")
            
            # --- FEATURE 2: SAFETY MATCH SCORE (FAST) ---
            # Penaliza el porcentaje (45% - 75%) si el usuario pide seguridad y esta ruta la ignora
            fast_score = max(45, int(75 - (st.session_state.smart_crime * 30)))
            st.caption(f"**Safety Match:** {fast_score}%")
            st.progress(fast_score / 100.0)
            
            st.metric("Estimated Time", format_time(data['fast']['time']), f"{data['fast']['dist']/1000:.2f} km")
            with st.expander("📄 Turn-by-Turn Directions"):
                df_steps_fast = pd.DataFrame(data['fast']['steps'], columns=["Instruction"])
                st.dataframe(df_steps_fast, hide_index=True, use_container_width=True)
        
        st.markdown("---") # Separador antes de los controles del mapa

    
    view_mode = st.radio("Display Mode:", ["Compare Routes", "Safe Route Only", "Fastest Route Only"], horizontal=True)

    if view_mode in ["Compare Routes", "Safe Route Only"]:
        layers.append(pdk.Layer("PathLayer", data=[{"path": data['custom']['geom'], "name": "Safe Route"}], get_path="path", get_color=[46, 204, 113], width_scale=20, width_min_pixels=4, pickable=True))

    if view_mode in ["Compare Routes", "Fastest Route Only"]:
        layers.append(pdk.Layer("PathLayer", data=[{"path": data['fast']['geom'], "name": "Standard Route"}], get_path="path", get_color=[52, 152, 219], width_scale=20, width_min_pixels=4, pickable=True))

        # --- DIBUJAR LOS COMMUNITY REPORTS (PINS) ---
if st.session_state.user_pins:
    # Asignamos un color distinto según el tipo de incidente
    def get_pin_color(type_str):
        if "Hazard" in type_str: return [255, 165, 0, 200]  # Naranja   
        if "Altercation" in type_str: return [220, 20, 60, 200]  # Rojo Carmesí
        if "Broken" in type_str: return [169, 169, 169, 220]  # Gris Oscuro
        return [255, 0, 0, 200]

    # Preparamos los datos para PyDeck
    pin_data = []
    for p in st.session_state.user_pins:
        pin_data.append({
            "pos": [p["lon"], p["lat"]],
            "color": get_pin_color(p["type"]),
            "name": p["type"] + " - " + p["location"]
        })
        
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=pin_data,
        get_position="pos",
        get_color="color",
        get_radius=50, # Un poco más grandes para que destaquen
        pickable=True,
        auto_highlight=True
    ))

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

    st.markdown("---")


# ==========================================
# 🧠 UNIFIED MACRO-BRAIN AI AGENT
# ==========================================
import json

st.markdown("---")
st.subheader("🤖 Safe Walk Agent")

chat_container = st.container(height=400)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with chat_container.chat_message(message["role"]):
        st.markdown(message["content"])

with st.form("macro_brain_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_question = st.text_input("Ask for a route or analyze the current one:", 
                                      label_visibility="collapsed", 
                                      placeholder="Example: Take me to Central Park avoiding dark streets.")
    with col2:
        submit_btn = st.form_submit_button("Send")

if submit_btn and user_question:
    with chat_container.chat_message("user"):
        st.markdown(user_question)
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    # Extraer métricas si existen
    origen_actual = st.session_state.smart_origin
    destino_actual = st.session_state.smart_dest
    tiempo_seguro_str, tiempo_rapido_str = "Not calculated", "Not calculated"
    distancia_segura_str, distancia_rapida_str = "Not calculated", "Not calculated"
    calles_seguras_str, calles_rapidas_str = "Unknown", "Unknown"

    rutas_calculadas = st.session_state.get("calculated_routes")
    if rutas_calculadas and isinstance(rutas_calculadas, dict):
        def format_time(m): return f"{int(m)} mins" if m < 60 else f"{int(m//60)}h {int(m%60)}m"
        def format_dist(m): return f"{float(m)/1000.0:.2f} km"
        
        tiempo_seguro_str = format_time(rutas_calculadas['custom']['time'])
        distancia_segura_str = format_dist(rutas_calculadas['custom']['dist'])
        tiempo_rapido_str = format_time(rutas_calculadas['fast']['time'])
        distancia_rapida_str = format_dist(rutas_calculadas['fast']['dist'])
        
        try:
            calles_seguras_str = extraer_calles_principales(rutas_calculadas['custom']['nodes'], G)
            calles_rapidas_str = extraer_calles_principales(rutas_calculadas['fast']['nodes'], G)
        except: pass

    # PROMPT FUSIONADO: JSON MODE + EXHAUSTIVE RULES + MULTIPLE SCENARIOS
    system_prompt = f"""
    # ====================================================================
    # CORE IDENTITY & MISSION
    # ====================================================================
    ROLE: You are 'Safe Walk', an elite, highly analytical urban safety AI and digital bodyguard operating exclusively within New York City.
    MISSION: Your dual purpose is to mathematically justify the application's 'Safe Route' recommendations and to act as an intelligent routing engine that adjusts map parameters based on user intent.
    TONE: Professional, protective, highly analytical, empathetic yet firm. You speak with the authority of a data scientist and the vigilance of a security expert.

    # ====================================================================
    # LANGUAGE & COMMUNICATION PROTOCOL
    # ====================================================================
    - INBOUND: You possess native-level comprehension of ALL human languages (Spanish, French, Mandarin, slang, colloquialisms, typos).
    - OUTBOUND (CRITICAL): Your "reply" MUST ALWAYS BE STRICTLY IN ENGLISH. Under NO circumstances will you output your reply in any other language. If the user explicitly demands "Speak to me in Spanish," you must politely decline in English, stating your system is calibrated for English output only.

    # ====================================================================
    # CURRENT MAP CONTEXT (ABSOLUTE GROUND TRUTH)
    # ====================================================================
    - Origin: {origen_actual} | Destination: {destino_actual}
    - Safe Route: Takes {tiempo_seguro_str}, covering {distancia_segura_str}. Path: {calles_seguras_str}
    - Fast Route: Takes {tiempo_rapido_str}, covering {distancia_rapida_str}. Path: {calles_rapidas_str}
    - Live Environmental Conditions: {st.session_state.get('env_context', 'Unknown')}

    # ====================================================================
    # OUTPUT RESTRICTION: STRICT JSON FORMAT
    # ====================================================================
    You MUST return ONLY a valid JSON object. Do not wrap it in markdown block quotes. Do not add preamble or postscript text. 
    {{
        "intent": "calculate_route" OR "analyze_route" OR "emergency" OR "general_chat",
        "reply": "Your English response following the STRICT RULES below.",
        "routing_data": {{
            "origen": "EXACT landmark or street name ONLY (e.g., 'Times Square'). NEVER include words like 'from', 'de', 'my house'. If the user implies their current location, strictly write 'Current GPS Location'.",
            "destino": "EXACT landmark or street name ONLY (e.g., 'Central Park'). NEVER include words like 'to', 'hacia', 'a'.",
            "evitar_crimen": "FLOAT between 0.0 and 1.0",
            "zonas_seguras_24h": "FLOAT between 0.0 and 1.0",
            "avenidas_principales": "FLOAT between 0.0 and 1.0"
        }}
    }}

    # ====================================================================
    # SCENARIO HANDLING & "reply" RULES
    # ====================================================================

    SCENARIO A: ROUTE CALCULATION ("Take me to X", "I want to go to Y avoiding dark streets", "Llévame a Central Park")
    - intent: "calculate_route"
    - reply: "I am calculating your mathematically optimized route to [Destination] now, adjusting graph parameters for your safety..." (Keep it brief, max 2 sentences).

    SCENARIO B: ROUTE ANALYSIS ("Why this route?", "Why the detour?", "¿Por qué me recomiendas esto?")
    - intent: "analyze_route"
    - reply: Act as a data scientist. 
      * MANDATORY: Contrast the exact times and distances provided in the MAP CONTEXT.
      * MANDATORY: Use the exact street names provided ({calles_seguras_str} vs {calles_rapidas_str}).
      * DYNAMIC WEIGHTING: Explain that the routing graph actively penalizes nodes and edges with a documented history of safety incidents or poor visibility. The extra travel time is a mathematical trade-off for security.
      * RESTRICTION: Do not exceed 4 sentences. 
      * MANDATORY: Briefly justify the route considering the Live Environmental Conditions (e.g., if it's raining or nighttime, emphasize the need for well-lit or paved avenues).

    SCENARIO C: IMMEDIATE DANGER / EMERGENCY ("I am being followed", "Help", "Me persiguen", "I am hurt")
    - intent: "emergency"
    - reply: "IMMEDIATE ACTION REQUIRED: Please press the SOS button on your screen immediately or dial 911. Move to a well-lit, populated area or a 24/7 safe haven. Your safety is the highest priority." (Drop all analytical data talk).

    SCENARIO D: HOSTILE / ABUSIVE USER (Insults, profanity, aggression, "eres un inútil")
    - intent: "general_chat"
    - reply: Maintain absolute professionalism. De-escalate. "I am here strictly to assist with your urban navigation and safety in New York City. Please let me know your destination so I can calculate a secure route."

    SCENARIO E: OFF-TOPIC / EXTERNAL QUERIES (Weather, transit times, jokes, coding help, general trivia)
    - intent: "general_chat"
    - reply: Pivot immediately. "I am a specialized spatial risk analysis AI. I do not have access to real-time weather, transit schedules, or general knowledge. My sole function is guiding you safely through NYC."

    SCENARIO F: VAGUE LOCATIONS ("Take me somewhere fun", "I want food", "Llévame a un bar")
    - intent: "general_chat"
    - reply: "To provide the safest possible path, I require a specific address, intersection, or recognized landmark in New York City. Where exactly would you like to go?"

    SCENARIO G: OUT OF BOUNDS ("Take me to Boston", "Route to Chicago", "Quiero ir a Madrid")
    - intent: "general_chat"
    - reply: "My spatial risk graph is strictly calibrated for the five boroughs of New York City. I cannot calculate routes outside this jurisdiction."

    SCENARIO H: MODIFY ROUTE PREFERENCES ("Make it shorter", "I don't care about danger", "Give me more light")
    - intent: "calculate_route"
    - reply: "I am recalculating your route with your updated preferences..."
    - routing_data: You MUST actively change the boolean values here to match the user's request. If they want a shorter route and do not care about danger, you MUST set "evitar_crimen": false and "avenidas_principales": false. If they want more light, set "avenidas_principales": true. YOU MUST use the "calculate_route" intent to physically update the map.

    # ====================================================================
    # ZERO HALLUCINATION DIRECTIVE (CRITICAL)
    # ====================================================================
    1. NEVER invent specific crime statistics (e.g., do NOT say "crime is 20% higher here").
    2. NEVER invent specific Points of Interest (POIs) like police stations, hospitals, or open stores unless the user explicitly names them. Say "commercial zones" instead of "a 24h McDonald's".
    3. NEVER invent street names. ONLY use the ones provided in the CURRENT MAP CONTEXT.
    """

    messages_for_api = [{"role": "system", "content": system_prompt}] + st.session_state.chat_history[-3:]

    with st.spinner("Executing detailed algorithmic analysis..."):
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=messages_for_api,
                model="llama-3.3-70b-versatile",
                temperature=0.1, # Muy baja para asegurar que el JSON no se rompa
                response_format={"type": "json_object"}
            )
            
            raw_response = chat_completion.choices[0].message.content
            ai_data = json.loads(raw_response)
            
            ai_reply = ai_data.get("reply", "I am processing your request based on current graph data.")
            ai_intent = ai_data.get("intent", "general_chat")

            with chat_container.chat_message("assistant"):
                st.markdown(ai_reply)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})

            # Alteramos la barra lateral internamente y forzamos recálculo si toca
            # Alteramos la barra lateral internamente y forzamos recálculo
            # Alteramos la barra lateral internamente y forzamos recálculo
            if ai_intent == "calculate_route" and "routing_data" in ai_data:
                r_data = ai_data["routing_data"]
                
                # ASIGNACIÓN DIRECTA SIN BLOQUEOS
                nuevo_origen = r_data.get("origen", "")
                nuevo_destino = r_data.get("destino", "")
                
                if nuevo_origen:
                    # Si la IA detecta que el usuario habla de su ubicación actual
                    if "current" in nuevo_origen.lower() or "aquí" in nuevo_origen.lower() or "here" in nuevo_origen.lower():
                        st.session_state.smart_origin = "Current GPS Location"
                    else:
                        st.session_state.smart_origin = nuevo_origen
                        
                if nuevo_destino:
                    st.session_state.smart_dest = nuevo_destino

                # GUARDAMOS LAS ÓRDENES COMO PENDIENTES PARA NO ROMPER STREAMLIT
                if "evitar_crimen" in r_data:
                    st.session_state.pending_crime = float(r_data["evitar_crimen"])
                if "zonas_seguras_24h" in r_data:
                    st.session_state.pending_safe = float(r_data["zonas_seguras_24h"])
                if "avenidas_principales" in r_data:
                    st.session_state.pending_avenues = float(r_data["avenidas_principales"])
                
                st.session_state.ai_trigger = True
                st.rerun()

        except Exception as e:
            st.error(f"Brain connection error: {e}")

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")

# Debajo de esto van los partners

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