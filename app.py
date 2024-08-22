import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

# Load the filtered dataset for Chennai
df = pd.read_csv('chennai_restaurants.csv')

# Streamlit app
st.title("Restaurant Recommendation System")

# Dropdown for area selection
areas = df['Area'].unique()
user_area = st.selectbox("Select Area", areas)

# Slider for price range
user_price = st.slider("Select Maximum Price", min_value=int(df['Price'].min()), max_value=int(df['Price'].max()), value=int(df['Price'].median()))

# Slider for rating
user_rating = st.slider("Select Minimum Rating", min_value=float(df['Avg ratings'].min()), max_value=float(df['Avg ratings'].max()), value=float(df['Avg ratings'].median()), step=0.1)

# Filter the dataset based on the user area
area_df = df[df['Area'] == user_area]

# Perform K-means clustering based on Price and Avg ratings
kmeans = KMeans(n_clusters=5, random_state=0)  # You can adjust n_clusters based on your needs
area_df['cluster'] = kmeans.fit_predict(area_df[['Price', 'Avg ratings']])

# Filter the results based on user's price and rating
filtered_df = area_df[(area_df['Price'] <= user_price) & (area_df['Avg ratings'] >= user_rating)]

# Display the top 5 restaurants that fit the criteria
top_restaurants = filtered_df.head(5)

# Output the results
st.subheader("Top 5 Restaurants")
if not top_restaurants.empty:
    st.write("Here are the top 5 restaurants based on your criteria:")
    for index, row in top_restaurants.iterrows():
        st.markdown(f"### {row['Restaurant']}")
        st.write(f"**Price:** ₹{row['Price']}")
        st.write(f"**Average Rating:** {row['Avg ratings']}")
        st.write(f"**Food Type:** {row['Food type']}")
        st.write(f"**Address:** {row['Address']}")
        st.write("---")
else:
    st.write("No restaurants found that match your criteria.")
