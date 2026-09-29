import streamlit as st
import pandas as pd
from datetime import datetime
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Configuración inicial de la página
st.set_page_config(page_title="ECO-DRAGA IA", page_icon="🌊", layout="wide")

# ----------------------------------------------------
# CARGA Y CACHÉ DEL MODELO DE IA
# ----------------------------------------------------
@st.cache_resource
def load_teachable_model():
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

try:
    model, class_names = load_teachable_model()
except Exception as e:
    st.error(f"Error cargando el modelo: {e}. Asegúrate de tener 'keras_model.h5' y 'labels.txt' en la misma carpeta.")

# ----------------------------------------------------
# MEMORIA DE REGISTRO EN SESIÓN
# ----------------------------------------------------
if "registro" not in st.session_state:
    st.session_state.registro = pd.DataFrame(columns=["Fecha", "Zona", "Tipo Residuo", "Cantidad Est.", "Prioridad"])

# ----------------------------------------------------
# NAVEGACIÓN - MENÚ LATERAL
# ----------------------------------------------------
st.sidebar.title("🤖 ECO-DRAGA IA")
st.sidebar.caption("Pantanos de Villa - Sistema Inteligente")

pantalla = st.sidebar.radio("Selecciona una Pantalla:", [
    "Inicio",
    "Escanear / Analizar",
    "Resultado",
    "Registro",
    "Prioridad"
])

# ----------------------------------------------------
# PANTALLA 1: INICIO
# ----------------------------------------------------
if pantalla == "Inicio":
    st.title("🌊 Sistema Mecatrónico Inteligente para Recolección de Residuos")
    st.subheader("Caso de aplicación: Pantanos de Villa")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Misión
        Crear una herramienta digital que aporte información útil para apoyar la identificación, registro y priorización de residuos sólidos flotantes.
        
        * **Base tecnológica:** Banda transportadora inteligente + Reconocimiento visual con IA.
        * **ODS Principales:** ODS 6 (Agua limpia), ODS 12 (Producción responsable), ODS 15 (Vida de ecosistemas).
        * **ODS Complementarios:** ODS 4, ODS 9, ODS 11.
        """)
    with col2:
        st.info("💡 **Indicación:** Navega en el menú lateral para realizar la identificación de residuos en tiempo real.")

# ----------------------------------------------------
# PANTALLA 2 Y 3: ESCANEAR / ANALIZAR Y RESULTADO
# ----------------------------------------------------
elif pantalla in ["Escanear / Analizar", "Resultado"]:
    st.title("📷 Escanear y Analizar Residuo Flotante")
    
    st.markdown("### Configuración de la Muestra")
    col_a, col_b = st.columns(2)
    with col_a:
        zona = st.selectbox("Selecciona la Zona de Pantanos de Villa:", [
            "Zona Laguna Mayor", 
            "Canal de Egreso", 
            "Zona Marvilla", 
            "Laguna Génesis"
        ])
    with col_b:
        cantidad_input = st.number_input("Cantidad aproximada base (referencial):", min_value=1, value=1, step=1)

    archivo_imagen = st.file_uploader("Carga una foto del residuo detectado:", type=["jpg", "png", "jpeg"])
    
    if archivo_imagen is not None:
        image = Image.open(archivo_imagen).convert("RGB")
        st.image(image, caption="Imagen seleccionada", width=300)
        
        if st.button("🔍 Ejecutar Análisis con IA"):
            # Preprocesamiento de imagen para Teachable Machine
            size = (224, 224)
            image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image_resized)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # Predicción del modelo
            prediction = model.predict(data)
            index = np.argmax(prediction)
            
            # Limpieza de etiqueta de Teachable Machine
            raw_class_name = class_names[index].strip()
            
            # Quitar números iniciales si existen
            partes = raw_class_name.split(' ', 1)
            if len(partes) > 1 and partes[0].isdigit():
                texto_clase = partes[1].strip()
            else:
                texto_clase = raw_class_name
            
            confianza = float(prediction[0][index]) * 100
            tipo_lower = texto_clase.lower().replace("_", " ")

            # --- EXTRAER SOLO EL MATERIAL PRINCIPAL ---
            if any(k in tipo_lower for k in ["plastico", "plástico", "plastic", "botella", "pet", "pvc"]):
                tipo_material = "Plástico"
            elif any(k in tipo_lower for k in ["papel", "paper", "carton", "cartón", "cardboard"]):
                tipo_material = "Papel"
            else:
                tipo_material = texto_clase.replace("_", " ").split()[0].capitalize()

            # --- AJUSTE AUTOMÁTICO DE CANTIDAD ESTIMADA MAYOR ---
            if "mucho" in tipo_lower:
                prioridad = "🔴 ALTA"
                # Rango de cantidad alta (ejemplo: 40 a 80 unidades)
                cant_calculada = max(int(cantidad_input), np.random.randint(40, 81))
            elif "poco" in tipo_lower:
                if any(k in tipo_lower for k in ["plastico", "plástico", "plastic", "botella", "pet", "pvc"]):
                    prioridad = "🟡 MEDIA"
                else:
                    prioridad = "🟢 BAJA"
                # Rango de cantidad baja (ejemplo: 2 a 6 unidades)
                cant_calculada = min(max(int(cantidad_input), 1), np.random.randint(2, 7))
            else:
                # Regla de respaldo
                cant_calculada = int(cantidad_input)
                if cant_calculada >= 20:
                    prioridad = "🔴 ALTA"
                elif cant_calculada >= 8:
                    prioridad = "🟡 MEDIA"
                else:
                    prioridad = "🟢 BAJA"

            # Guardar resultado en sesión
            st.session_state["resultado_actual"] = {
                "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Zona": zona,
                "Tipo Residuo": tipo_material,
                "Confianza": f"{confianza:.1f}%",
                "Cantidad Est.": cant_calculada,
                "Prioridad": prioridad
            }
            st.success("Análisis finalizado con éxito.")

    # Muestra de Pantalla 3: Resultado
    if "resultado_actual" in st.session_state:
        st.markdown("---")
        st.header("📊 RESULTADO DEL ANÁLISIS")
        res = st.session_state["resultado_actual"]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("RESIDUO DETECTADO", res["Tipo Residuo"], delta=f"Precisión: {res['Confianza']}")
        m2.metric("NIVEL DE PRIORIDAD", res["Prioridad"])
        m3.metric("CANTIDAD / ZONA", f"{res['Cantidad Est.']} unids. en {res['Zona']}")
        
        if st.button("💾 Registrar en la Base de Datos"):
            nuevo = pd.DataFrame([{
                "Fecha": res["Fecha"],
                "Zona": res["Zona"],
                "Tipo Residuo": res["Tipo Residuo"],
                "Cantidad Est.": res["Cantidad Est."],
                "Prioridad": res["Prioridad"]
            }])
            st.session_state.registro = pd.concat([st.session_state.registro, nuevo], ignore_index=True)
            st.success("✅ Registro guardado exitosamente.")

# ----------------------------------------------------
# PANTALLA 4: REGISTRO
# ----------------------------------------------------
elif pantalla == "Registro":
    st.title("📋REGISTRO HISTÓRICO DE RESIDUOS")
    st.caption("Estructura requerida: Fecha | Zona | Tipo | Cantidad | Prioridad")
    
    if st.session_state.registro.empty:
        st.info("Aún no se han guardado análisis. Ve a la 'Pantalla 2' para escanear muestras.")
    else:
        st.dataframe(st.session_state.registro, use_container_width=True)
        
        csv_data = st.session_state.registro.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar Reporte en CSV", csv_data, "registro_pantanos_villa.csv", "text/csv")

# ----------------------------------------------------
# PANTALLA 5: PRIORIDAD
# ----------------------------------------------------
elif pantalla == "Prioridad":
    st.title("🚨 CRITERIOS Y TOMA DE DECISIONES")
    st.markdown("""
    ### Respuesta a la Pregunta Fundamental del Jurado:
    > **¿Qué decisión puede tomar una persona utilizando la información que proporciona nuestra aplicación?**
    
    La aplicación evalúa el nivel de acumulación y el tipo de residuo según las categorías detectadas:
    
    * 🔴 **PRIORIDAD ALTA:** 
      * **Criterio:** Categoría detectada con acumulación alta ("mucho", ~40-80 unidades).
      * **Decisión:** Despliegue prioritario y encendido de la banda transportadora Eco-Draga en la zona asignada.
    * 🟡 **PRIORIDAD MEDIA:** 
      * **Criterio:** Presencia baja de plástico o botellas ("plástico poco", ~2-6 unidades).
      * **Decisión:** Programación de recolección en el turno correspondiente.
    * 🟢 **PRIORIDAD BAJA:** 
      * **Criterio:** Presencia baja de materiales biodeteriorables como papel o cartón ("papel poco", ~2-6 unidades).
      * **Decisión:** Monitoreo o recolección manual de rutina.
    """)