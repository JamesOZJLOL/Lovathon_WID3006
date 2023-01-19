import streamlit as st

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

# st.markdown(
#     """
#     <style>
#     [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
#         width: 500px;
#     }
#     [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
#         width: 500px;
#         margin-left: -500px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

model = CatBoostRegressor()
model.load_model('001.model')

age = st.sidebar.number_input("What is your age?")
height = st.sidebar.number_input("What is your height? (cm)")
weight = st.sidebar.number_input("What is your weight? (kg)")
siblings = st.sidebar.radio("Do you have any opposite gender siblings?", ("Yes", "No"))


appearance = st.sidebar.radio("From a scale of 1-5, rate your own physical appearance.", ("1","2","3","4","5"))
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

pets = st.sidebar.radio("How much do you like having pets?", ("1","2","3","4","5"))
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

travel = st.sidebar.radio("How much do you like to travel?", ("1","2","3","4","5"))
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

financially = st.sidebar.radio("How comfortable are you financially?" , ("1","2","3","4","5"))
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

feeling = st.sidebar.radio("I always know others' feeling.", ("1","2","3","4","5"))
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

children =st.sidebar.radio("How much do you like children?", ("1","2","3","4","5"))
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

anxious_ppl = st.sidebar.radio("I feel anxious when meeting new people.", ("1","2","3","4","5"))
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

alone = st.sidebar.radio("I would rather spend my weekend time alone rather than with other people", ("1","2","3","4","5"))
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

silent = st.sidebar.radio("Even if I'm right in the discussion, I stay silent to not hurt others.", ("1","2","3","4","5"))
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

true = st.sidebar.radio("I believe that true love DO exist.", ("1","2","3","4","5"))
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

aware = st.sidebar.radio("I am aware of my appearance.", ("1","2","3","4","5"))
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

eq = st.sidebar.radio("I believe that I have a good Emotional Quotient (EQ)", ("1","2","3","4","5"))
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

iq = st.sidebar.radio("I believe that I have a good Intelligence Quotient (IQ)", ("1","2","3","4","5"))
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


def run_model():
    df = pd.DataFrame({
        'height': height, 
        'weight': weight, 
        'siblings': siblings, 
        'appearance': appearance, 
        'pets': pets, 
        'travel': travel, 
        'financially': financially, 
        'feeling': feeling, 
        'children': children, 
        'anxious_ppl': anxious_ppl, 
        'alone': alone, 
        'silent': silent, 
        'true': true, 
        'aware': aware, 
        'eq': eq, 
        'iq': iq, 
    }, index=[0])

    df.iloc[:,:2] = (df.iloc[:,:2] - np.array([174.43912176, 62.26546906])) / np.array([102.43791061, 103.50437648])

    df['siblings'] = df['siblings'].map({'Yes':1, 'No':0})
    df = df.astype('int32')

    df = pd.get_dummies(df, columns=[
        'siblings', 
        'appearance', 
        'pets', 
        'travel', 
        'financially', 
        'feeling', 
        'children', 
        'anxious_ppl', 
        'alone', 
        'silent', 
        'true', 
        'aware', 
        'eq', 
        'iq',
    ])

    df2 = pd.DataFrame(columns=['height', 'weight', 'siblings_0', 'siblings_1', 'appearance_1',
        'appearance_2', 'appearance_3', 'appearance_4', 'appearance_5',
        'pets_1', 'pets_2', 'pets_3', 'pets_4', 'pets_5', 'travel_1',
        'travel_2', 'travel_3', 'travel_4', 'travel_5', 'financially_1',
        'financially_2', 'financially_3', 'financially_4', 'financially_5',
        'feeling_1', 'feeling_2', 'feeling_3', 'feeling_4', 'feeling_5',
        'children_1', 'children_2', 'children_3', 'children_4', 'children_5',
        'anxious_ppl_1', 'anxious_ppl_2', 'anxious_ppl_3', 'anxious_ppl_4',
        'anxious_ppl_5', 'alone_1', 'alone_2', 'alone_3', 'alone_4', 'alone_5',
        'silent_1', 'silent_2', 'silent_3', 'silent_4', 'silent_5', 'true_1',
        'true_2', 'true_3', 'true_4', 'true_5', 'aware_1', 'aware_2', 'aware_3',
        'aware_4', 'aware_5', 'eq_1', 'eq_2', 'eq_3', 'eq_4', 'eq_5', 'iq_1',
        'iq_2', 'iq_3', 'iq_4', 'iq_5'])
    
    df = pd.concat([df2, df]).fillna(0.0)


    y = model.predict(df)[0]

    output = 'soon!' if age >= y else f'in {int((y-age).round(0))} years!'


    st.markdown("<h1 style='display: flex; justify-content: center; color: white;'>❤️✨</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='display: flex; justify-content: center; color: white;'>You will find your true</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='display: flex; justify-content: center; color: white;'>love {output}</h1>", unsafe_allow_html=True)


if (height < 60 or height > 300):
    st.sidebar.write("Height must be between 60 and 300!")
elif (weight <30 or weight > 200):
    st.sidebar.write("Weight must be between 30 and 200!")
elif st.sidebar.button("Submit"):
    run_model()


# st.markdown("<h1 style='text-align: center; color: white;'>years! </h1>", unsafe_allow_html=True)