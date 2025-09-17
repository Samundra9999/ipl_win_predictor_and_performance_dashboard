import pandas as pd
import numpy as np
import pickle
import streamlit as st
import plotly.express as px
import os
import requests

st.markdown("""
    <style>
    /* Tabs container */
    .stTabs [data-baseweb="tab-list"] {
        gap: 22rem;
        background-color: #0D0C0C;
        padding: 0.5rem;
        border-radius: 10px;
    }
    

    /* Default (inactive) tabs */
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
        color: #444;
        background-color: #FFFFFF;
        opacity: 1 !important;
    }

    /* Active (selected) tab */
    .stTabs [aria-selected="true"] {
        background-color: #FF7043; 
        color: white !important;
        font-weight: 700;
        border: 2px solid #105a82;
    }
    </style>
""", unsafe_allow_html=True)

tab1,tab2 = st.tabs(['IPL win probability prediction','Player Status'])

with tab1:

    teams = ['Delhi Capitals', 'Sunrisers Hyderabad',
        'Royal Challengers Bengaluru', 'Kolkata Knight Riders',
        'Chennai Super Kings', 'Mumbai Indians', 'Lucknow Super Giants',
        'Rajasthan Royals', 'Gujarat Titans', 'Punjab Kings']    

    cities = ['Bengaluru', 'Pune', 'Jaipur', 'Mumbai', 'Chennai', 'Centurion',
        'Bangalore', 'Delhi', 'Abu Dhabi', 'Kolkata', 'Raipur',
        'Navi Mumbai', 'East London', 'Ranchi', 'Cuttack', 'Hyderabad',
        'Bloemfontein', 'Dharamsala', 'Port Elizabeth', 'Chandigarh',
        'Durban', 'Kimberley', 'Ahmedabad', 'Dubai', 'Cape Town',
        'Guwahati', 'Indore', 'Visakhapatnam', 'Lucknow', 'Johannesburg',
        'Nagpur', 'Sharjah', 'Mohali']


    pipe = pickle.load(open('pipe.pkl','rb'))


    st.title('IPL win probability prediction')

    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox('Select the batting team',sorted(teams))
    with col2:
        bowling_team = st.selectbox('Select the bowling team',sorted(teams))

    selected_city = st.selectbox('Select host city',sorted(cities))

    target = st.number_input('Target')

    col3,col4,col5 = st.columns(3)

    with col3:
        score = st.number_input('Score')
    with col4:
        overs = st.number_input('Overs completed')
    with col5:
        wickets = st.number_input('Wickets out')

    if st.button('Predict Probability'):
        st.session_state.white_bg = True
        runs_left = target - score
        balls_left = 120 - (overs*6)
        wickets = 10 - wickets
        crr = score/overs
        rrr = (runs_left*6)/balls_left

        input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'target_runs':[target],'required_run':[runs_left],'balls_left':[balls_left],'wicket_left':[wickets],'current_rr':[crr],'required_rr':[rrr]})

        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]
        st.header(batting_team + "- " + str(round(win*100)) + ""
        "%")
        st.header(bowling_team + "- " + str(round(loss*100)) + "%")

        
with tab2:
    st.title('Player Status')   
    url = "https://drive.google.com/uc?id=1LmDd7_zb_dMlxKfBYxf1GAISyZbVA4kS"
    local_path = "data/deliveries.csv"

    if not os.path.exists(local_path):
        response = requests.get(url)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download file, status code: {response.status_code}")
    else:
        print("File already exists.")

    df = pd.read_csv(local_path)    
    Batter = df['batter'].unique().tolist()
    Bowler = df['bowler'].unique().tolist()

    col1, col2 = st.columns(2)

    with col1:
        selected_batter = st.selectbox('Select the Batter', Batter)
    with col2:
        selected_bowler = st.selectbox('Select the Bowler', Bowler)

    if st.button('Compare Players'):
        data = df[(df['batter']==selected_batter)&(df['bowler']==selected_bowler)]
        if data.empty:
            st.warning("No data found for this batter-bowler combination!")
        else:
            runs = data['batsman_runs'].value_counts().reset_index().sort_values('batsman_runs')
            total_runs = data['batsman_runs'].sum()
            balls = data.shape[0]
            strike_rate = total_runs*100/balls
            try:
                dismissals = data['player_dismissed'].value_counts().reset_index().iloc[0,1]
            except:
                dismissals = 0

            st.text('Balls faced'+ " -  " + str(balls))
            st.text('Runs Scored by Batter'+ " - "+str(total_runs))
            st.text('Strike_rate'+ " - " + str(round(strike_rate,2)))
            st.text('Dismissals'+ " -  " + str(dismissals))


            colors = {0:'gray',1:'lightblue',2:'blue',4:'yellow',6:'red'}

            fig = px.pie(runs, values='count', names='batsman_runs',
                color='batsman_runs', 
                color_discrete_map=colors,
                hole=0.4)
            fig.update_layout(title="Run Distribution")
            st.plotly_chart(fig)
            