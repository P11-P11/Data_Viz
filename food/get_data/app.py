import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import json
import io

# ------------------------------
# Set Page Configuration
# ------------------------------
st.set_page_config(page_title="🍴 Interactive Ingredient Recommender 🍴", layout="wide")

# ------------------------------
# Load Preprocessed Data
# ------------------------------
@st.cache_data
def load_data(serialized=True):
    """
    Load preprocessed data from CSV files and NearestNeighbors model from pickle.
    
    Parameters:
        serialized (bool): Whether to load the serialized ingredients.
    
    Returns:
        tuple: (clean_recipes_df, merged_embedding_df, nbrs)
    """
    try:
        if serialized:
            # Load serialized ingredients
            clean_recipes_df = pd.read_csv('clean_recipes_df_serialized.csv')
            # Deserialize the ingredients column
            clean_recipes_df['ingredients'] = clean_recipes_df['ingredients'].apply(json.loads)
        else:
            # Load ingredients as lists directly (if already saved correctly)
            clean_recipes_df = pd.read_csv('clean_recipes_df.csv')
            # Parse the ingredients column if necessary
            import ast
            clean_recipes_df['ingredients'] = clean_recipes_df['ingredients'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        
        clean_embedding_df = pd.read_csv(r'C:\Users\ezequ\OneDrive\Documentos\Facultad\data viz\food\get_data\clean_embedding_df.csv')
        
        # Merge clean_embedding_df with clean_recipes_df to include 'id'
        merged_embedding_df = clean_embedding_df.merge(
            clean_recipes_df[['id']],  # Removed 'name' column as it does not exist
            left_index=True,
            right_index=True,
            how='left'
        )
        
        with open('nearest_neighbors.pkl', 'rb') as f:
            nbrs = pickle.load(f)
        
        return clean_recipes_df, merged_embedding_df, nbrs
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        st.stop()

# Load data
clean_recipes_df, merged_embedding_df, nbrs = load_data(serialized=True)

# ------------------------------
# Sample 10 Random Recipes
# ------------------------------
@st.cache_data
def get_random_recipes(df, num_samples=10):
    """
    Select random recipes from the DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing recipes.
        num_samples (int): Number of random samples to select.
        
    Returns:
        pd.DataFrame: DataFrame containing random recipes.
    """
    return df.sample(n=num_samples, random_state=None).reset_index(drop=True)

# Initialize session state for random recipes and favorites
if 'random_recipes' not in st.session_state:
    st.session_state.random_recipes = get_random_recipes(clean_recipes_df, num_samples=10)
if 'favorites' not in st.session_state:
    st.session_state.favorites = []

def refresh_recipes():
    """
    Refresh the list of random recipes.
    """
    st.session_state.random_recipes = get_random_recipes(clean_recipes_df, num_samples=10)

# ------------------------------
# Recommend Ingredients Function
# ------------------------------
def find_neighbors(nbrs, embedding_df, recipe_index):
    """
    Find nearest neighbors for a given recipe index.
    
    Parameters:
        nbrs (NearestNeighbors): Fitted NearestNeighbors model.
        embedding_df (pd.DataFrame): DataFrame containing UMAP1 and UMAP2.
        recipe_index (int): Index of the target recipe.
        
    Returns:
        list: Indices of the nearest neighbors.
    """
    distances, indices = nbrs.kneighbors([embedding_df.loc[recipe_index, ['UMAP1', 'UMAP2']]])
    return indices.flatten()

def recommend_ingredients(recipe_id, clean_recipes_df, merged_embedding_df, nbrs, top_k=5, threshold=0.5):
    """
    Recommend ingredient additions and removals for a given recipe.
    
    Parameters:
        recipe_id (int): ID of the target recipe.
        clean_recipes_df (pd.DataFrame): DataFrame with filtered and cleaned recipes.
        merged_embedding_df (pd.DataFrame): DataFrame containing UMAP embeddings and 'id'.
        nbrs (NearestNeighbors): Fitted NearestNeighbors model.
        top_k (int): Number of recommendations for additions and removals.
        threshold (float): Frequency threshold for recommending removals.
        
    Returns:
        dict: Recommendations containing additions and removals.
    """
    # Locate the recipe index
    recipe_indices = clean_recipes_df.index[clean_recipes_df['id'] == recipe_id].tolist()
    if not recipe_indices:
        st.warning(f"Recipe ID {recipe_id} not found.")
        return None
    recipe_index = recipe_indices[0]
    
    # Find nearest neighbors
    neighbors = find_neighbors(nbrs, merged_embedding_df, recipe_index)
    
    # Get ingredients of the target recipe
    target_ingredients = set(clean_recipes_df.loc[recipe_index, 'ingredients'])
    
    # Aggregate ingredients from neighbors
    neighbor_ingredients = []
    for neighbor in neighbors:
        if neighbor != recipe_index:
            neighbor_ingredients.extend(clean_recipes_df.loc[neighbor, 'ingredients'])
    neighbor_ingredients = set(neighbor_ingredients)
    
    # Identify potential additions (ingredients in neighbors but not in target)
    potential_additions = neighbor_ingredients - target_ingredients
    
    # Identify potential removals (ingredients in target but less common in neighbors)
    neighbor_ingredients_list = []
    for neighbor in neighbors:
        if neighbor != recipe_index:
            neighbor_ingredients_list.extend(clean_recipes_df.loc[neighbor, 'ingredients'])
    ingredient_counts = Counter(neighbor_ingredients_list)
    
    removal_candidates = []
    for ingredient in target_ingredients:
        if ingredient_counts[ingredient] < (len(neighbors) * threshold):
            removal_candidates.append(ingredient)
    
    # Recommend top_k additions based on frequency
    additions = Counter()
    for ingredient in potential_additions:
        additions[ingredient] = ingredient_counts[ingredient]
    top_additions = [item for item, count in additions.most_common(top_k)]
    
    # Recommend top_k removals
    removals = removal_candidates[:top_k]
    
    return {
        'recipe_id': recipe_id,
        'additions': top_additions,
        'removals': removals
    }

# ------------------------------
# Visualize Recommendations
# ------------------------------
def plot_umap(recipe_id, clean_recipes_df, merged_embedding_df, nbrs, selected_key):
    """
    Plot the UMAP embedding highlighting the target recipe and its neighbors.

    Parameters:
        recipe_id (int): ID of the target recipe.
        clean_recipes_df (pd.DataFrame): DataFrame with filtered and cleaned recipes.
        merged_embedding_df (pd.DataFrame): DataFrame containing UMAP embeddings and 'id'.
        nbrs (NearestNeighbors): Fitted NearestNeighbors model.
        selected_key (str): Identifier for the UMAP embedding parameters.
    """
    # Locate the recipe index
    recipe_indices = clean_recipes_df.index[clean_recipes_df['id'] == recipe_id].tolist()
    if not recipe_indices:
        st.warning(f"Recipe ID {recipe_id} not found.")
        return
    recipe_index = recipe_indices[0]

    # Find nearest neighbors
    neighbors = find_neighbors(nbrs, merged_embedding_df, recipe_index)

    # Prepare data for Plotly
    plot_df = merged_embedding_df.copy()
    plot_df['type'] = 'Other Recipes'
    plot_df.loc[neighbors, 'type'] = 'Nearest Neighbors'
    plot_df.loc[recipe_index, 'type'] = 'Target Recipe'

    color_discrete_map = {
        'Other Recipes': 'blue',          # Color for other recipes
        'Nearest Neighbors': 'red'        # Color for nearest neighbors
    }

    fig = px.scatter(
        plot_df,
        x='UMAP1',
        y='UMAP2',
        color='type',
        color_discrete_map=color_discrete_map,  # Apply the custom color map
        hover_data=['id'],                      # Display recipe ID on hover
        title=f'UMAP Projection with Recommendations for Recipe ID {recipe_id} ({selected_key})',
        labels={'UMAP1': 'UMAP Dimension 1', 'UMAP2': 'UMAP Dimension 2'},
        width=800,
        height=600
)
    # Update marker sizes and symbols for all traces except the target
    fig.update_traces(marker=dict(size=7, opacity=0.6))

    # Highlight the target recipe using go.Scatter()
    target_recipe = plot_df.loc[[recipe_index]]
    fig.add_trace(
        go.Scatter(
            x=target_recipe['UMAP1'],
            y=target_recipe['UMAP2'],
            mode='markers',
            marker=dict(color='pink', size=12, symbol='star'),
            name='Target Recipe',
            hovertext=target_recipe['id'],
            hoverinfo='text'
        )
    )

    # Display the interactive Plotly chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Display recipe details below the plot
    st.markdown(f"### 📄 Recipe Details: ID {recipe_id}")
    st.write(f"**Cuisine:** {clean_recipes_df.loc[recipe_index, 'cuisine']}")
    st.write(f"**Ingredients:** {', '.join(clean_recipes_df.loc[recipe_index, 'ingredients'])}")
    if 'instructions' in clean_recipes_df.columns and pd.notna(clean_recipes_df.loc[recipe_index, 'instructions']):
        st.write(f"**Instructions:** {clean_recipes_df.loc[recipe_index, 'instructions']}")
    if 'url' in clean_recipes_df.columns and pd.notna(clean_recipes_df.loc[recipe_index, 'url']):
        st.markdown(f"[View Full Recipe]({clean_recipes_df.loc[recipe_index, 'url']})")

# ------------------------------
# Streamlit App Layout
# ------------------------------
def main():
    st.title("🍴 **Interactive Ingredient Recommender** 🍴")
    
    # Sidebar for search, filters, and actions
    with st.sidebar:
        st.header("🔍 **Search & Filters** 🔍")
        
        # Radio buttons to choose between random selection and arbitrary ID input
        selection_mode = st.radio(
            "🗂️ Select Recipe Mode:",
            ("Select from Random Recipes", "Enter Recipe ID")
        )
        
        # Recommendation Settings
        st.header("⚙️ **Recommendation Settings** ⚙️")
        top_k = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)
        threshold = st.slider("Removal Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        
        # Refresh Recipes Button (only visible in random selection mode)
        if selection_mode == "Select from Random Recipes":
            if st.button('🔄 Refresh Recipes'):
                refresh_recipes()
                st.success("Recipes refreshed!")
        
        # Initialize selected_recipe_id
        selected_recipe_id = None
        
        if selection_mode == "Select from Random Recipes":
            # Dropdown with 10 random recipes
            recipe_options = st.session_state.random_recipes['id'].tolist()
            selected_recipe_id = st.selectbox(
                "🔽 Select a Recipe:",
                options=recipe_options,
                format_func=lambda x: f"Recipe ID {x}"
            )
        else:
            # Enter Recipe ID
            entered_id = st.text_input("✏️ Enter Recipe ID:")
            if entered_id:
                # Validate if the entered ID is an integer
                if entered_id.isdigit():
                    entered_id = int(entered_id)
                    if entered_id in clean_recipes_df['id'].values:
                        selected_recipe_id = entered_id
                    else:
                        st.warning("Recipe ID not found in the dataset.")
                else:
                    st.warning("Please enter a valid integer Recipe ID.")
        
        # Display all 10 random recipes for reference (only in random selection mode)
        if selection_mode == "Select from Random Recipes":
            st.header("📚 **Random Recipes** 📚")
            # Show a dataframe with recipe details
            display_recipes = st.session_state.random_recipes[['id', 'cuisine', 'ingredients']].copy()
            # Convert ingredients list to comma-separated string for better display
            display_recipes['ingredients'] = display_recipes['ingredients'].apply(lambda x: ', '.join(x))
            st.dataframe(display_recipes, height=300)
        
        # Favorites Section
        st.header("🌟 **Your Favorite Recipes** 🌟")
        if st.session_state.favorites:
            favorite_recipes = clean_recipes_df[clean_recipes_df['id'].isin(st.session_state.favorites)][['id', 'cuisine']]
            favorite_recipes['ingredients'] = favorite_recipes['id'].apply(lambda x: ', '.join(clean_recipes_df.loc[clean_recipes_df['id'] == x, 'ingredients'].values[0]))
            st.dataframe(favorite_recipes, height=200)
        else:
            st.write("You have no favorite recipes yet.")
        
        # Add to Favorites Button
        if selected_recipe_id and st.button("⭐ Add to Favorites"):
            if selected_recipe_id not in st.session_state.favorites:
                st.session_state.favorites.append(selected_recipe_id)
                st.success("Recipe added to favorites!")
            else:
                st.info("Recipe is already in favorites.")

    # Tabs for organized content
    tabs = st.tabs(["📄 Details", "🔍 Recommendations", "📈 Visualization", "📝 Feedback", "ℹ️ About"])

    with tabs[0]:
        st.header(f"📄 **Recipe Details: ID {selected_recipe_id}** 📄")
        
        if selected_recipe_id:
            # Fetch recipe details
            recipe_details = clean_recipes_df[clean_recipes_df['id'] == selected_recipe_id]
            if recipe_details.empty:
                st.error("Selected Recipe ID not found in the dataset.")
                return
            recipe_details = recipe_details.iloc[0]
            
            # Display Cuisine and Ingredients
            col1, col2 = st.columns([1, 3])
            with col1:
                st.subheader("🌍 **Cuisine:**")
            with col2:
                st.write(recipe_details['cuisine'])
            
            with col1:
                st.subheader("📝 **Ingredients:**")
            with col2:
                st.write(", ".join(recipe_details['ingredients']))
            
            # Display Instructions and URL if available
            if 'instructions' in recipe_details and pd.notna(recipe_details['instructions']):
                with col1:
                    st.subheader("📖 **Instructions:**")
                with col2:
                    st.write(recipe_details['instructions'])
            
            if 'url' in recipe_details and pd.notna(recipe_details['url']):
                with col1:
                    st.subheader("🔗 **Recipe Link:**")
                with col2:
                    st.markdown(f"[View Full Recipe]({recipe_details['url']})")
        else:
            st.info("Please select or enter a recipe to view details.")

    with tabs[1]:
        st.header("🔍 **Ingredient Recommendations** 🔍")
        
        if selected_recipe_id:
            # Generate Recommendations
            with st.spinner('Generating ingredient recommendations...'):
                recommendations = recommend_ingredients(
                    recipe_id=selected_recipe_id,
                    clean_recipes_df=clean_recipes_df,
                    merged_embedding_df=merged_embedding_df,  # Pass the merged DataFrame
                    nbrs=nbrs,
                    top_k=top_k,
                    threshold=threshold
                )
            
            if recommendations:
                # Fetch recipe details
                recipe_details = clean_recipes_df[clean_recipes_df['id'] == selected_recipe_id].iloc[0]
                
                # Display Current Ingredients
                st.subheader("📝 **Current Ingredients:**")
                st.write(", ".join(recipe_details['ingredients']))
                
                # Recommendations to Add
                st.subheader("✅ **Ingredients to Add:**")
                if recommendations['additions']:
                    additions = "\n".join([f"• {ingredient}" for ingredient in recommendations['additions']])
                    st.markdown(additions)
                else:
                    st.write("No strong recommendations for additions.")
                
                # Recommendations to Remove
                st.subheader("❌ **Ingredients to Remove:**")
                if recommendations['removals']:
                    removals = "\n".join([f"• {ingredient}" for ingredient in recommendations['removals']])
                    st.markdown(removals)
                else:
                    st.write("No strong recommendations for removals.")
                
                # Download Recommendations
                st.subheader("📥 **Download Recommendations** 📥")
                recommendation_data = {
                    'recipe_id': recommendations['recipe_id'],
                    'ingredients_to_add': recommendations['additions'],
                    'ingredients_to_remove': recommendations['removals']
                }
                st.download_button(
                    label="📥 Download as JSON",
                    data=json.dumps(recommendation_data, indent=4),
                    file_name=f'recommendations_recipe_{selected_recipe_id}.json',
                    mime='application/json'
                )
                
                # Additional Download Options
                # Convert recommendations to DataFrame
                rec_df = pd.DataFrame({
                    'Ingredients to Add': recommendations['additions'],
                    'Ingredients to Remove': recommendations['removals']
                })
                
                # CSV Download
                csv = rec_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f'recommendations_recipe_{selected_recipe_id}.csv',
                    mime='text/csv'
                )
                
                # Excel Download
                excel_buffer = io.BytesIO()
                rec_df.to_excel(excel_buffer, index=False)
                st.download_button(
                    label="📥 Download as Excel",
                    data=excel_buffer.getvalue(),
                    file_name=f'recommendations_recipe_{selected_recipe_id}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            else:
                st.info("No recommendations available for the selected recipe.")
        else:
            st.info("Please select or enter a recipe to view recommendations.")

    with tabs[2]:
        st.header("📈 **UMAP Visualization** 📈")
        
        if selected_recipe_id:
            selected_key = 'nn_45_md_0.15'  # Placeholder for actual key or parameter
            plot_umap(selected_recipe_id, clean_recipes_df, merged_embedding_df, nbrs, selected_key)
            
            # Additional Charts
            # Ingredient Frequency Bar Chart
            st.subheader("📊 **Ingredient Frequency Among Neighbors** 📊")
            recipe_indices = clean_recipes_df.index[clean_recipes_df['id'] == selected_recipe_id].tolist()
            if recipe_indices:
                recipe_index = recipe_indices[0]
                neighbors = find_neighbors(nbrs, merged_embedding_df, recipe_index)
                ingredient_freq = Counter()
                for neighbor in neighbors:
                    if neighbor != recipe_index:
                        ingredient_freq.update(clean_recipes_df.loc[neighbor, 'ingredients'])
                freq_df = pd.DataFrame(ingredient_freq.most_common(10), columns=['Ingredient', 'Count'])
                fig_freq = px.bar(freq_df, x='Ingredient', y='Count', title='Top 10 Ingredients Among Neighbors')
                st.plotly_chart(fig_freq, use_container_width=True)
            else:
                st.warning("No neighbors found to analyze ingredients.")
            
            # Cuisine Distribution Pie Chart
            st.subheader("🥘 **Cuisine Distribution Among Neighbors** 🥘")
            if recipe_indices:
                cuisine_counts = clean_recipes_df.loc[neighbors, 'cuisine'].value_counts().reset_index()
                cuisine_counts.columns = ['Cuisine', 'Count']
                fig_cuisine = px.pie(cuisine_counts, names='Cuisine', values='Count', title='Cuisine Distribution')
                st.plotly_chart(fig_cuisine, use_container_width=True)
            else:
                st.warning("No neighbors found to analyze cuisines.")
        else:
            st.info("Please select or enter a recipe to view the visualization.")

    with tabs[3]:
        st.header("📝 **Your Feedback** 📝")
        
        if selected_recipe_id:
            feedback = st.radio(
                "How useful are these recommendations?",
                ('Very Useful', 'Somewhat Useful', 'Not Useful')
            )
            if st.button("Submit Feedback"):
                # Log feedback to a file (append mode)
                try:
                    with open('feedback.log', 'a') as f:
                        f.write(f"Recipe ID {selected_recipe_id}: {feedback}\n")
                    st.success("Thank you for your feedback!")
                except Exception as e:
                    st.error(f"Failed to save feedback: {e}")
        else:
            st.info("Please select or enter a recipe to provide feedback.")

    with tabs[4]:
        st.header("ℹ️ **About This App** ℹ️")
        st.write("""
            **Interactive Ingredient Recommender** helps you optimize your recipes by suggesting ingredients to add or remove based on similar recipes in our database. 

            **Features:**
            - **Search and Filter:** Easily find recipes by ID or ingredient and filter by cuisine.
            - **Recommendations:** Get tailored suggestions to enhance or simplify your recipes.
            - **Visualization:** Explore recipe relationships through interactive UMAP plots.
            - **Feedback:** Share your thoughts to help improve the recommendation system.
            - **Favorites:** Save your favorite recipes for quick access later.

            **How It Works:**
            The app leverages UMAP for dimensionality reduction and a Nearest Neighbors model to find similar recipes. Based on the analysis, it provides recommendations for ingredient additions and removals to optimize your culinary creations.

            **Developed with ❤️ using Streamlit.**
        """)
    
    # Add an expander for additional information or tips
    with st.expander("ℹ️ **How to Use This App**"):
        st.write("""
            1. **Select Recipe Mode:** Choose between selecting from random recipes or entering an arbitrary Recipe ID.
            2. **Search & Filters:** Use the search bar to find recipes by ID or ingredient and filter by cuisine.
            3. **View Details:** Click on a recipe to view its details, including cuisine and ingredients.
            4. **Get Recommendations:** Navigate to the "Recommendations" tab to see suggested ingredients to add or remove.
            5. **Explore Visualization:** Use the "Visualization" tab to explore the relationships between recipes and analyze ingredient frequency and cuisine distribution among neighbors.
            6. **Provide Feedback:** Share your feedback in the "Feedback" tab to help improve the system.
            7. **Manage Favorites:** Add recipes to your favorites in the sidebar for easy access later.
        """)

    # Add a footer
    st.markdown("---")
    st.write("Developed with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
