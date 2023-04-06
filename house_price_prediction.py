import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns 
import matplotlib.pylab as plt
import pickle


st.title("Tashkent House Price Prediction")

house_df = pd.read_csv("cleaned_tashkent_house.csv")

st.subheader(f"Data Frame consists of {house_df.shape[0]} rows and {house_df.shape[1]} columns. ")
st.write("Data about houses sold in 2021")

st.write("Here is the first 20 rows of the dataset:")

st.table(house_df.head(20))

grouped_by_district = pd.DataFrame(house_df.groupby('district')["price"].mean())
grouped_by_rooms = pd.DataFrame(house_df.groupby('rooms')["price"].mean())


st.header("Let's do some data analysis")

st.write("Some of data analysis on the district column")

fig, ax = plt.subplots(1,2)
fig.tight_layout()

sns.countplot(data=house_df, x="district", ax=ax[0])
grouped_by_district.plot(kind="bar", ax=ax[1])
ax[0].set_title("number of sold houses in each district", fontsize=9)
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
ax[0].set_yticklabels(ax[0].get_yticklabels(), rotation=45)

ax[1].set_title("Average price of houses in each district", fontsize=9)
ax[1].set_yticklabels(ax[1].get_yticklabels(), rotation=45)


st.pyplot(fig)

st.write("Some of data analysis on the room column")

fig2, ax2 = plt.subplots(1,2)
fig2.tight_layout(pad=3)

grouped_by_rooms.plot(kind="bar", ax=ax2[0])
sns.countplot(data=house_df, x="rooms", ax=ax2[1])

ax2[0].set_title("Average price of houses based on rooms", size=8)
ax2[0].set_xticklabels(ax2[0].get_xticklabels(), fontsize=7)
ax2[0].set_yticklabels(ax2[0].get_yticklabels(), fontsize=7)
ax2[0].set_ylabel("Price", fontsize=8)


ax2[1].set_xticklabels(ax2[1].get_xticklabels(), fontsize=7)
ax2[1].set_yticklabels(ax2[1].get_yticklabels(), fontsize=7)
ax2[1].set_title("Number of the same Nō rooms", size=8)
ax2[1].set_ylabel("Count", fontsize=8)

st.pyplot(fig2)

st.write("Correlation:")

fig3, ax3 = plt.subplots(2,1, figsize=(16,10))
fig3.tight_layout()

sns.scatterplot(data=house_df, x="size", y="price", hue="rooms", ax=ax3[0])
sns.scatterplot(data=house_df, x="size", y="price", hue="district", ax=ax3[1])


st.pyplot(fig3)

st.subheader("Conclusion:")
st.write("We have seen that the most expensive houses are in Mirobod district and the cheapest houses are in Bektemir district.")
st.write("Moreover in the bar graph we can see that in 2021 in Chilonzor district houses were sold more than other districts.")
st.write("Moreover as the number of rooms increase, house price increases which make sense.")
st.write("And in scatter plot we can that there is a positive correlation between price and size which make sense. As the size of house increases, price increase.")


st.subheader("Below you can see approximate price of house you want to buy or sell.")

district = st.selectbox(
    'Choose the district where you would like to buy or sell a house',
    tuple(house_df["district"].unique()))

size_of_house = st.number_input('Enter the size of house(in square meter)', min_value=20, max_value=300)
flat_num = st.number_input('Enter the floor number', min_value=1, max_value=15)
max_num = st.number_input('Enter the maximum floor of house', min_value=1, max_value=15)
num_of_rooms = st.number_input('Enter the number of rooms', min_value=1, max_value=15)

button = st.button("Calculate price")

pipe = pickle.load(open("RFM_House_price_prediction.pkl", "rb"))

if button:
    new_df = pd.DataFrame([[district, num_of_rooms, size_of_house, flat_num, max_num]], columns=["district","rooms","size","level","max_levels"])
    prediction = pipe.predict(new_df)
    st.write(f"Price of house is {round(prediction[0])}$")

st.subheader("Contact me:")
st.write("github: https://github.com/Abduqayyum")
st.write("email: rabduqayum@mail.ru")
st.write("linkedin: linkedin.com/in/abduqayum-rasulmukhamedov-70844624a")



# option = st.selectbox(
#     'How would you like to be contacted?',
#     ('Email', 'Home phone', 'Mobile phone'))

# st.write('You selected:', option)
