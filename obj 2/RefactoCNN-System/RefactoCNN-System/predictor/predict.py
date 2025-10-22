
def run_prediction(parsed_data):
    for filename, tree in parsed_data:
        print(f"Analyzing {filename}...")
        
        print("Prediction: Suggest 'Extract Method' for long blocks.")
