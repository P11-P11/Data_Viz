import streamlit as st
import json
import numpy as np
import plotly.graph_objects as go
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from ollama import embeddings as ollama_embeddings
from langchain_ollama.llms import OllamaLLM  # Import the correct LLM for text generation

# ------------------------------
# Set Page Configuration
# ------------------------------
st.set_page_config(page_title="🧠 Interactive RAG App", layout="wide")

# ------------------------------
# Load Vector Database
# ------------------------------
@st.cache_data
def load_vector_db(json_file):
    """
    Load vector database from a JSON file.
    
    Parameters:
        json_file (str): Path to the JSON file containing the vector database.
        
    Returns:
        dict: Dictionary containing vector embeddings and metadata.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        vector_db = json.load(f)
    return vector_db

# Load the vector DB (example path)
vector_db = load_vector_db(r"C:\Users\ezequ\OneDrive\Documentos\Facultad\data viz\rag\star wars\vector_db.json")

# Extract embeddings and metadata from the loaded DB
embeddings = []
contents = []
titles = []
movies = []

for key, value in vector_db.items():
    embeddings.append(value['embedding'])
    contents.append(value['content'])
    titles.append(value['metadata']['title'])
    movies.append(value['metadata']['movie'])

# Convert embeddings list into a NumPy array for further processing
embeddings = np.array(embeddings)

# Standardize the embeddings for retrieval
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# ------------------------------
# Function to Perform Retrieval
# ------------------------------
def retrieve_documents(query, embeddings_scaled, umap_embeddings, use_umap=True):
    """
    Retrieve documents using the original or UMAP-reduced embeddings.
    
    Parameters:
        query (str): User query for document retrieval.
        embeddings_scaled (np.array): Scaled embeddings for high-dimensional retrieval.
        umap_embeddings (np.array): UMAP-reduced embeddings.
        use_umap (bool): Whether to use UMAP embeddings for retrieval.
        
    Returns:
        list: Retrieved document indices.
    """
    # Embed the query
    query_embedding_response = ollama_embeddings(model="mxbai-embed-large", prompt=query)
    query_embedding = query_embedding_response['embedding']
    
    # Standardize the query embedding
    query_embedding_scaled = scaler.transform([query_embedding])
    
    # Choose which embeddings to use
    if use_umap:
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(umap_embeddings)
        distances, indices = nbrs.kneighbors(umap_reducer.transform(query_embedding_scaled))
    else:
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(embeddings_scaled)
        distances, indices = nbrs.kneighbors(query_embedding_scaled)
    
    return indices.flatten()

# ------------------------------
# UMAP Dimensionality Reduction
# ------------------------------
umap_reducer = umap.UMAP(n_neighbors=9, n_components=3, min_dist=0.5, random_state=42)
umap_embeddings = umap_reducer.fit_transform(embeddings_scaled)

# ------------------------------
# Streamlit App Layout
# ------------------------------
st.title("🧠 **Interactive RAG App**")

# Text Input for Query
query = st.text_input("Enter your query here:", value="The Force will be with you. Always.")

if query:

    # Retrieval without UMAP and with UMAP in two columns
    st.subheader("Retrieved Documents")

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    retrieved_with_umap = retrieve_documents(query, embeddings_scaled, umap_embeddings, use_umap=True)
    retrieved_without_umap = retrieve_documents(query, embeddings_scaled, umap_embeddings, use_umap=False)

    with col1:
        st.write("**Documents Retrieved Without UMAP:**")
        for i, idx in enumerate(retrieved_without_umap):
            st.write(f"**Document {i+1}:** {titles[idx]}")
            st.write(contents[idx][:200] + "...")  # Display only the first 200 characters

    with col2:
        st.write("**Documents Retrieved With UMAP:**")
        for i, idx in enumerate(retrieved_with_umap):
            st.write(f"**Document {i+1}:** {titles[idx]}")
            st.write(contents[idx][:200] + "...")  # Display only the first 200 characters



    # ------------------------------
    # Plot UMAP Embeddings
    # ------------------------------
    st.subheader("📈 **UMAP Visualization**")
    
    query_embedding_3d = umap_reducer.transform(scaler.transform([ollama_embeddings(model="mxbai-embed-large", prompt=query)['embedding']]))


    # Step 1: Create a color map for movies
    unique_movies = list(set(movies))  # Get a list of unique movie names
    color_map = {movie: idx for idx, movie in enumerate(unique_movies)}  # Assign a unique color index to each movie
    colors = [color_map[movie] for movie in movies]  # Map each chunk to a color based on its movie

    # Step 2: Plot UMAP embeddings with Plotly, using numerical colors for movies
    trace_movies = go.Scatter3d(
        x=umap_embeddings[:, 0],
        y=umap_embeddings[:, 1],
        z=umap_embeddings[:, 2],
        mode='markers',
        marker=dict(size=5, opacity=0.6, color=colors, colorscale='Viridis'),
        text=titles,
        name="Movies"
    )

    # Plot the query vector as a distinct shape (e.g., diamond)
    trace_query = go.Scatter3d(
        x=[query_embedding_3d[0, 0]],
        y=[query_embedding_3d[0, 1]],
        z=[query_embedding_3d[0, 2]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name="Query Vector"
    )

    # Plot the nearest neighbors as distinct points
    trace_neighbors = go.Scatter3d(
        x=umap_embeddings[retrieved_with_umap, 0],
        y=umap_embeddings[retrieved_with_umap, 1],
        z=umap_embeddings[retrieved_with_umap, 2],
        mode='markers',
        marker=dict(size=8, color='green', symbol='cross'),
        name="Nearest Neighbors"
    )

    # Step 3: Create the plot figure
    fig = go.Figure(data=[trace_movies, trace_query, trace_neighbors])

    # Step 4: Update layout and show the plot
    fig.update_layout(
        title="3D UMAP Projection of Document Embeddings with Query and Nearest Neighbors",
        scene=dict(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            zaxis_title='UMAP Dimension 3'
        )
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)


    # ------------------------------
    # LLM Generation with Retrieved Content
    # ------------------------------
    st.subheader("📝 **Generated Responses Using Retrieved Content**")

    # Initialize the LLM model (only once)
    llm = OllamaLLM(model='llama3.2:1b')

    # Prepare the system prompt
    system_prompt = (
        "You are a Jedi historian and scholar of the Force, entrusted with documenting and interpreting the teachings, trials, and wisdom of the Jedi Order. "
        "Your duty is to analyze and explain the mystical ways of the Force in guiding the Jedi through their journeys, while being precise, concise, and insightful.\n"
        "Use the context provided below and answer in no more than four sentences, distilling the essence of the Force's influence over Luke's journey.\n\n"
    )



    # Combine retrieved document content into a single string for both cases
    retrieved_content_without_umap = "\n\n".join([contents[idx] for idx in retrieved_without_umap])
    retrieved_content_with_umap = "\n\n".join([contents[idx] for idx in retrieved_with_umap])

    # Construct the input prompts
    input_text_without_umap = f"{system_prompt}Context:\n{retrieved_content_without_umap}\n\nQuestion: {query}"
    input_text_with_umap = f"{system_prompt}Context:\n{retrieved_content_with_umap}\n\nQuestion: {query}"

    print(input_text_without_umap)
    print()
    print(input_text_with_umap)

    # Generate responses using the LLM
    response_without_umap = llm.invoke(input_text_without_umap)
    response_with_umap = llm.invoke(input_text_with_umap)

    # Display the responses side by side
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Response without UMAP:**")
        st.write(response_without_umap)

    with col2:
        st.write("**Response with UMAP:**")
        st.write(response_with_umap)
