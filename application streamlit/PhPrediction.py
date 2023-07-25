import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium as fl
from streamlit_folium import st_folium
import firebase_admin

from firebase_admin import credentials
from firebase_admin import auth
from firebase_admin import firestore

cred = credentials.Certificate('projet1-dd682-bc859e41a042.json')
#firebase_admin.initialize_app(cred)


# 4 variables
PHSPHT_SILICAT_OXYGENE_TCARBN = pickle.load(open("model/allFeatures/XGBOOST_all_features.pkl", "rb"))

# 3 VARIABLES

PHSPHT_TCARBN_OXYGEN = pickle.load(open("model/comb3v/XGBoost_3v_phspht_tcarbn_oxygen_", "rb"))
NITRAT_PHSPHT_TCARBN = pickle.load(open("model/comb3v/XGBoost_3v_nitrat_phspht_tcarbn_", "rb"))
NITRAT_PHSPHT_OXYGEN = pickle.load(open("model/comb3v/XGBoost_3v_nitrat_phspht_oxygen_", "rb"))
NITRAT_TCARBN_OXYGEN = pickle.load(open("model/comb3v/XGBoost_3v_nitrat_tcarbn_oxygen_", "rb"))

# 2 VARIABLES

PHSPHT_OXYGEN = pickle.load(open("model/comb2v/XGBoost_2v_phspht_oxygen_", "rb"))
PHSPHT_TCARBN = pickle.load(open("model/comb2v/XGBoost_2v_phspht_tcarbn_", "rb"))
NITRAT_OXYGEN = pickle.load(open("model/comb2v/XGBoost_2v_nitrat_oxygen_", "rb"))
NITRAT_PHSPHT = pickle.load(open("model/comb2v/XGBoost_2v_nitrat_phspht_", "rb"))
NITRAT_TCARBN=pickle.load(open("model/comb2v/XGBoost_2v_nitrat_tcarbn_", "rb"))
TCARBN_OXYGEN = pickle.load(open("model/comb2v/XGBoost_2v_tcarbn_oxygen_", "rb"))


# 1 VARIABLE

OXYGENE_model = pickle.load(open("model/oneFeature/XGBoost_1v_oxygen_", "rb"))
PHOSPHT_model = pickle.load(open("model/oneFeature/XGBoost_1v_phspht_", "rb"))
NITRAT_model = pickle.load(open("model/oneFeature/XGBoost_1v_nitrat_", "rb"))
TCARBN_model = pickle.load(open("model/oneFeature/XGBoost_1v_tcarbn_", "rb"))


# reshape des valeurs saisie par l'utilisateur
def features(float_features):
    ft = list(map(float, float_features))
    fts = np.array(ft)
    fts = fts.reshape(1, -1)
    return fts


def get_pos(lng, lat):
    return lat, lng


# firebase_admin.initialize_app()
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'useremail' not in st.session_state:
    st.session_state.useremail = ''


def f():
    try:
        user = auth.get_user_by_email(email)
        st.success("Connexion réussie!")
        st.session_state.username = user.uid
        st.session_state.useremail = user.email
        st.session_state.signedout = True
        st.session_state.signout = True

    except:
        st.warning("Erreur de connexion ")


def t():
    st.session_state.signout = False
    st.session_state.signedout = False
    st.session_state.username = ''


def addMarker(m, mapData):
    return m.add_child(
        fl.Marker(
            location=[mapData.Latitude, mapData.Longitude],
            popup=
            "Latitude: " + str(mapData.Latitude) + "<br>"
            + "Longitude: " + str(mapData.Longitude) + "<br>"
            + "pH: " + str(mapData.pH)
        ).add_to(m)
    )


def ph():
    col1, col2, col3 = st.columns(3)
    nbVr = []
    prediction = 0
    with col1:
        if st.checkbox('Nitrat'):
            nbVr.append(1)
            nit = st.number_input('Nitrat')
    with col2:
        if st.checkbox('Oxygène'):
            nbVr.append(2)
            oxy = st.number_input('Oxygène')
        if st.checkbox('Totale Carbonique'):
            nbVr.append(3)
            tCarbn = st.number_input('TCARBN')
    with col3:
        if st.checkbox('Phosphate'):
            nbVr.append(4)
            pho = st.number_input('Phosphate')
    models = ("Machine learning", "Deep learning")
    models_index = st.sidebar.selectbox("Select a model", range(
        len(models)), format_func=lambda x: models[x], key=1)




    # marker = None
    if st.button("Prédire ph"):
        m = fl.Map(location=[0, -170], zoom_start=2)
        m.add_child(fl.LatLngPopup())
        ##si l'utilasateur a choisi une seul variable
        if (len(nbVr) == 1):
            if (nbVr[0] == 1):
                float_features = [nit]
                prediction = NITRAT_model.predict(features(float_features))


            elif (nbVr[0] == 2):
                float_features = [oxy]
                prediction = OXYGENE_model.predict(features(float_features))

            elif (nbVr[0] == 3):
                float_features = [tCarbn]
                prediction = TCARBN_model.predict(features(float_features))
            elif (nbVr[0] == 4):
                float_features = [pho]
                prediction = PHOSPHT_model.predict(features(float_features))
        # Deux variables
        if len(nbVr) == 2:
            if (nbVr == [1, 2]):
                df = pd.DataFrame(data={'Nitrat': nit, 'OXYGEN ': oxy}, index=[0])
                df = np.array(df)
                prediction = NITRAT_OXYGEN.predict(df)

            elif (nbVr == [1, 3]):
                df = pd.DataFrame(data={'Nitrat': nit, 'Tcarbn': tCarbn}, index=[0])
                df = np.array(df)
                prediction = NITRAT_TCARBN.predict(df)

            elif (nbVr == [1, 4]):
                df = pd.DataFrame(data={'NITRAT': tCarbn, 'PHSPHAT': pho}, index=[0])
                df = np.array(df)
                prediction = NITRAT_PHSPHT.predict(df)

            elif (nbVr == [2, 4]):
                df = pd.DataFrame(data={'OXYGEN': oxy, 'PHSPHT': pho}, index=[0])
                df = np.array(df)
                prediction = PHSPHT_OXYGEN.predict(df)

            elif (nbVr == [2, 3]):
                df = pd.DataFrame(data={'OXYGEN': oxy, 'TCARBN': tCarbn}, index=[0])
                df = np.array(df)
                prediction = TCARBN_OXYGEN.predict(df)

            elif (nbVr == [3, 4]):
                df = pd.DataFrame(data={'TCARBN': tCarbn, 'PHSPHT': pho}, index=[0])
                df = np.array(df)
                prediction = PHSPHT_TCARBN.predict(df)

        if len(nbVr) == 3:
             if (nbVr == [1, 3, 4]):
                df = pd.DataFrame(data={'NITRAT': nit, 'TCARBN': tCarbn, 'PHSPHT': pho}, index=[0])
                df = np.array(df)
                prediction = NITRAT_PHSPHT_TCARBN.predict(df)
             elif (nbVr == [2, 3, 4]):
                df = pd.DataFrame(data={'OXYGEN': oxy, 'TCARBN': tCarbn, 'PHSPHT': pho}, index=[0])
                df = np.array(df)
                prediction = PHSPHT_TCARBN_OXYGEN.predict(df)

             elif (nbVr == [1, 2, 4]):
                df = pd.DataFrame(data={'NITRAT': nit, 'OXYGEN': oxy, 'PHSPHT': pho}, index=[0])
                df = np.array(df)
                prediction = NITRAT_PHSPHT_OXYGEN.predict(df)
             elif (nbVr == [1, 2, 3]):
                df = pd.DataFrame(data={'NITRAT': nit, 'OXYGEN': oxy, 'TCARBN': tCarbn}, index=[0])
                df = np.array(df)
                prediction = NITRAT_TCARBN_OXYGEN.predict(df)

        if len(nbVr) == 4:
            df = pd.DataFrame(data={'NITRAT': nit, 'OXYGEN': oxy, 'TCARBN': tCarbn,'PHSPHT':pho}, index=[0])
            df = np.array(df)
            prediction= PHSPHT_SILICAT_OXYGENE_TCARBN.predict(df)



        print("###########", nbVr)
        print(prediction)
        st.success(float(prediction))

        # Enregistrer dans Firebase
        db = firestore.client()
        user_collection = db.collection(st.session_state.username)
        new_document = user_collection.document()
        data = {
            "latitude": lat,
            "longitude": lng,
            "profondeur": depth,
            "ph": float(prediction),
            "date": firestore.SERVER_TIMESTAMP
        }
        new_document.set(data)











if 'signedout' not in st.session_state:
    st.session_state.signedout = False
if 'signout' not in st.session_state:
    st.session_state.signout = False
if not st.session_state['signedout']:
    choice = st.selectbox('login/signup', ['login', 'signup'])
    if choice == 'login':
        email = st.text_input('Email adresse')
        password = st.text_input('password', type='password')
        st.button("Se connecter", on_click=f)



    else:

        username = st.text_input('Entrer your unique username')
        email = st.text_input('Email Adresse')
        password = st.text_input('password', type='password')
        if st.button('Create my account'):
            user = auth.create_user(email=email, password=password, uid=username)
            st.success('account created successfully')
            st.markdown('Please login using your email ans password')
if st.session_state.signout:
    st.title("prédiction de Ph des oceans")
    st.text('name : ' + st.session_state.username)
    st.text('Email : ' + st.session_state.useremail)
    depth = st.slider('Depth', 0, 7000, 25)
    lat = st.number_input("Latitude")
    lng = st.number_input("Longitude")
    m = fl.Map(location=[0, -170], zoom_start=2)
    fl.TileLayer('Stamen Watercolor').add_to(m)
    fl.Marker(location=[lat, lng],
              popup=f"(latitude:{lat}, longtitude:{lng}, depth:{depth}m)",
              ).add_to(m)
    st_folium(m, height=500, width=800)
    ph()
    if st.button("historique"):
        m = fl.Map(tiles="Stamen Watercolor")
        m.add_child(fl.LatLngPopup())
        db = firestore.client()
        docs = db.collection(st.session_state.username).stream()
        predictions = []
        for doc in docs:
            predictions.append([

                doc.to_dict()["latitude"],
                doc.to_dict()["longitude"],
                doc.to_dict()["profondeur"],
                doc.to_dict()["ph"],
                doc.to_dict()["date"]
            ])

        predictions_df = pd.DataFrame(
            predictions,
            columns=["Latitude", "Longitude", "profondeur", "pH", "Date"]
        )
        st.table(predictions_df)

    st.button('sign out ', on_click=t)





