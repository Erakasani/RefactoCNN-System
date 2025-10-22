
from preprocessing.ast_parser import parse_java_files
from predictor.predict import run_prediction

if __name__ == "__main__":
    java_dir = "sample_java"
    parsed_data = parse_java_files(java_dir)
    run_prediction(parsed_data)
