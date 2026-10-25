import streamlit as st
import numpy as np
import pandas as pd
import base64
import random
import pickle
import joblib

st.set_page_config(page_title="Octobot", page_icon='media/iconNationAi.png', initial_sidebar_state='collapsed', layout='wide')
st.logo('media/Logo.png', size='large')

if 'model' not in st.session_state:
    # loading ML model
    with open("model.pkl", 'rb') as f:
        st.session_state.model = pickle.load(f)
    st.session_state.scaler = joblib.load('scaler.pkl')

# st.header(":rainbow[Task 3 - Nasa Hazardous Objects Detection Simulation]", text_alignment='center')
def set_bg_video(video_file):
  with open(video_file, "rb") as f:
    encoded_video = base64.b64encode(f.read()).decode()

  html_code = f"""
    <style>
    MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    # header {{visibility: hidden;}}

    .video-background {{
        position: fixed;
        left: 0;
        top: 0;
        min-width: 100%;
        min-height: 100%;
        z-index: -1;
        object-fit: cover;
    }}

    .stApp {{
        background: transparent !important;
    }}
    </style>

    <video autoplay muted loop class="video-background">
        <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
        Error: your browser does't support loading background video!
    </video>
    """
  st.markdown(html_code, unsafe_allow_html=True)

# set_bg_video(r"media\space.webm")
set_bg_video("media/Video Project 21.webm")

def set_local_background(image_file):
  with open(image_file, "rb") as file:
    encoded_string = base64.b64encode(file.read()).decode()

  st.markdown(
      f"""
         <style>
         .stApp {{
             background-image: url("data:image/png;base64,{encoded_string}");
             background-size: cover;
             background-position: center;
             background-repeat: no-repeat;
         }}
         </style>
         """,
      unsafe_allow_html=True,
  )

if 'df' not in st.session_state:
    df = pd.read_csv(r"data/nasa-nearest-earth-objects(1910-2024)-dataset.csv")
    df.dropna(inplace=True)
    df.drop('neo_id', axis=1, inplace=True)
    df.drop('orbiting_body', axis=1, inplace=True)
    df.is_hazardous = df.is_hazardous.astype(int)
    # df.drop('orbiting_body', axis=1, inplace=True)
    df['estimated_diameter_min'] = (df['estimated_diameter_min']+df['estimated_diameter_max'])/2
    df.drop("estimated_diameter_max", axis=1, inplace=True)
    df.columns = ['name','absolute_magnitude', 'estimated_diameter_average', 'relative_velocity', 'miss_distance','is_hazardous']
    # df.drop("absolute_magnitude", axis=1, inplace=True)
    st.session_state.df = df

if 'sample' not in st.session_state:
    st.session_state.sample = st.session_state.df[['name','absolute_magnitude','estimated_diameter_average','relative_velocity', 'miss_distance']].sample(1).to_dict('list')
    st.session_state.is_new = False
# s = np.array([st.session_state.sample[i] for i in ['estimated_diameter_average', 'relative_velocity', 'miss_distance']]).reshape(1,3)
# st.write(list(s))

def predict():
   # st.session_state.response = random.randint(0,1)
   s = np.array([st.session_state.sample[i] for i in ['estimated_diameter_average', 'relative_velocity', 'miss_distance']]).reshape(1,3)
   s = st.session_state.scaler.transform(s)
   st.session_state.response = st.session_state.model.predict(s)
   
def add_sample(sample):
    # name, magnitude, diameter_average, velocity, distance
    st.session_state.sample = {'name': [sample[0]],
                               'absolute_magnitude': [sample[1]],
                               'estimated_diameter_average': [sample[2]],
                               'relative_velocity': [sample[3]],
                               'miss_distance': [sample[4]]}
    predict()


if 'response' not in st.session_state:
    st.session_state.response = predict()
with st.spinner("Loading Simulation..."):
    #                                                                     space1,   moon,  distance1, earth, distance2, comet, space2
    space1, moon, distance1, earth, distance2, comet, space2 = st.columns([ 0.85,    0.4,     0.6,      1,     0.4,      0.3,   1.1 ])
    with moon:
        st.image("media/moon.png", caption='Moon')
        st.image('media/distance.png', width='stretch', caption='3,475 km')
    with distance1:
        st.space('xsmall')
        st.image('media/distance.png', width='stretch', caption='384,400 km')
    with earth:
        st.image("media/earth.gif", width='stretch', caption='Earth')
        st.image('media/distance.png', width='stretch', caption='12,742 km')
    with distance2:
        st.space('xsmall')
        st.image('media/distance.png', width='stretch', caption=f"{round(st.session_state.sample['miss_distance'][0],3)} km")
    with comet:
        st.image("media/object.png", caption=st.session_state.sample['name'][0][st.session_state.sample['name'][0].index('('):] if '(' in st.session_state.sample['name'][0] else st.session_state.sample['name'][0] )
        st.image('media/distance.png', width='stretch', caption=f"{round(st.session_state.sample['estimated_diameter_average'][0], 3)} km")
        st.caption(f":material/rocket_launch: Velocity  \n{round(st.session_state.sample['relative_velocity'][0],3)} km/h", text_alignment='center')


if 'sample_space' not in st.session_state:
    df = st.session_state.df
    st.session_state.sample_space = pd.concat([df[df["is_hazardous"]==True].sample(10), df[df["is_hazardous"]==False].sample(10)]).reset_index(drop=True)

with space2:
    with st.popover("Try a sample"):
        st.subheader("Try a Sample", divider='blue')
        names = st.session_state.sample_space['name'].values
        name = st.selectbox('Object Name',names)
        if name:
            s = st.session_state.sample_space[st.session_state.sample_space['name']==name]
            # st.write(name)
            # for i in s:
            #    st.write(str(i) + ': ' + str(s[i]))
        if st.button("Predict", type='primary', key=111):
            add_sample(s[['name','absolute_magnitude','estimated_diameter_average','relative_velocity', 'miss_distance']].values[0])
            st.session_state.is_new = False
            st.rerun()


# st.table(sample)
with space1:
    with st.popover(":material/add: Insert new object"):
        st.subheader(":material/add: Insert new object", divider='blue')
        name = st.text_input("Objectc Name:", type='default', value='162117 (1998 SD15)')
        magnitude = st.number_input("Magnitude:", value=19.14)
        # estimated_diameter_min	estimated_diameter_max	orbiting_body	relative_velocity	miss_distance	is_hazardou
        diameter_average = st.number_input("Diameter Average:", value=0.6390615)
        # diameter_max = st.number_input("Diameter Max:", value=0.883161)
        velocity = st.number_input("Velocity:", value=71745.401048)
        distance = st.number_input("Distance:", value=5.814362e+07)
        if st.button("Predict", type='primary'):
           add_sample([name, magnitude, diameter_average, velocity, distance])
           st.session_state.is_new = True
           st.rerun()


if st.session_state.response == 1:
       st.success(f"""Object "{st.session_state.sample['name'][0]}" is Safe""")
else:
    st.error(f"""Warning: Object"{st.session_state.sample['name'][0]}" is dangerous!""")