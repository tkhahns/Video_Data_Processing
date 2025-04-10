import pandas as pd
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import HfApi, HfHubHTTPError

# Initialize Hugging Face API
api = HfApi()

# Path to your CSV file
csv_file_path = r'models_reference_list.csv'

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Extract the "Name of the Model" column
model_names = df['Name of the Model'].dropna().unique()

# Function to check and fetch models from Hugging Face
def fetch_model_from_huggingface(model_name):
    try:
        # Search for the model on Hugging Face
        search_results = api.list_models(search=model_name)
        if search_results:
            print(f"Model '{model_name}' found on Hugging Face!")
            # Download the first matching model
            model_id = search_results[0].modelId
            model = AutoModel.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            print(f"Downloaded model: {model_id}")
        else:
            print(f"Model '{model_name}' not found on Hugging Face.")
    except HfHubHTTPError as e:
        print(f"Error fetching model '{model_name}': {e}")

# Iterate through the model names and fetch them
for model_name in model_names:
    print(f"Searching for model: {model_name}")
    fetch_model_from_huggingface(model_name)