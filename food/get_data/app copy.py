import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import json
import plotly.graph_objects as go  # Import graph_objects

# ------------------------------
# Set Page Configuration
# ------------------------------
# This must be the first Streamlit command in the script
st.set_page_config(page_title="Interactive Ingredient Recommender", layout="wide")

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
        
        clean_embedding_df = pd.read_csv('clean_embedding_df.csv')
        
        # Merge clean_embedding_df with clean_recipes_df to include 'id'
        merged_embedding_df = clean_embedding_df.merge(
            clean_recipes_df[['id']],
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

# Initialize session state for random recipes
if 'random_recipes' not in st.session_state:
    st.session_state.random_recipes = get_random_recipes(clean_recipes_df, num_samples=10)

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

    # Create Plotly scatter plot using Plotly Express
    fig = px.scatter(
        plot_df,
        x='UMAP1',
        y='UMAP2',
        color='type',
        hover_data=['id'],
        title=f'UMAP Projection with Recommendations for Recipe ID {recipe_id} ({selected_key})',
        labels={'UMAP1': 'UMAP Dimension 1', 'UMAP2': 'UMAP Dimension 2'},
        width=800,
        height=600
    )

    # Update marker sizes and symbols
    fig.update_traces(marker=dict(size=7, opacity=0.6))

    # Highlight the target recipe using go.Scatter()
    target_recipe = plot_df.loc[[recipe_index]]
    fig.add_trace(
        go.Scatter(
            x=target_recipe['UMAP1'],
            y=target_recipe['UMAP2'],
            mode='markers',
            marker=dict(color='red', size=12, symbol='star'),
            name='Target Recipe',
            hovertext=target_recipe['id'],
            hoverinfo='text'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Streamlit App Layout
# ------------------------------
def main():
    st.title("🍴 **Interactive Ingredient Recommender** 🍴")
    
    # Sidebar for selection and actions
    with st.sidebar:
        st.header("🔍 **Recipe Selection** 🔍")
        
        # Button to refresh random recipes
        if st.button('🔄 Refresh Recipes'):
            refresh_recipes()
            st.success("Recipes refreshed!")
        
        # Dropdown with 10 random recipes
        recipe_options = st.session_state.random_recipes['id'].tolist()
        selected_recipe_id = st.selectbox(
            "Choose a Recipe ID:",
            options=recipe_options,
            format_func=lambda x: f"Recipe ID {x}"
        )
        
        # Display all 10 random recipes for reference
        st.header("📚 **Random Recipes** 📚")
        # Show a dataframe with recipe details
        display_recipes = st.session_state.random_recipes[['id', 'cuisine', 'ingredients']]
        # Convert ingredients list to comma-separated string for better display
        display_recipes['ingredients'] = display_recipes['ingredients'].apply(lambda x: ', '.join(x))
        st.dataframe(display_recipes, height=300)
    
    # Main content area
    st.header(f"📄 **Recipe Details: ID {selected_recipe_id}** 📄")
    
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
    
    # Generate Recommendations
    with st.spinner('Generating ingredient recommendations...'):
        recommendations = recommend_ingredients(
            recipe_id=selected_recipe_id,
            clean_recipes_df=clean_recipes_df,
            merged_embedding_df=merged_embedding_df,  # Pass the merged DataFrame
            nbrs=nbrs,
            top_k=5,
            threshold=0.5
        )
    
    if recommendations:
        st.header("🔍 **Ingredient Recommendations** 🔍")
        
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
        
        # Visualization
        st.header("📈 **UMAP Visualization** 📈")
        selected_key = 'nn_45_md_0.15'
        plot_umap(selected_recipe_id, clean_recipes_df, merged_embedding_df, nbrs, selected_key)
    
    # Optional: Download Recommendations
    if recommendations:
        recommendation_data = {
            'recipe_id': recommendations['recipe_id'],
            'ingredients_to_add': recommendations['additions'],
            'ingredients_to_remove': recommendations['removals']
        }
        st.download_button(
            label="📥 Download Recommendations as JSON",
            data=json.dumps(recommendation_data, indent=4),
            file_name=f'recommendations_recipe_{selected_recipe_id}.json',
            mime='application/json'
        )
    
    # Footer or additional info
    st.markdown("---")
    st.write("Developed with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
