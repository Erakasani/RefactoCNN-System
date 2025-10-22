
import argparse
from preprocessing.ast_parser import parse_java_files
from predictor.predict import run_prediction

def main():
    parser = argparse.ArgumentParser(description="RefactoCNN CLI Tool")
    parser.add_argument("input_path", type=str, help=r"C:\Users\krake\Downloads\lakshmi prasana\obj 2\RefactoCNN-System\RefactoCNN-System\sample_java")
    args = parser.parse_args()
    
    parsed_data = parse_java_files(args.input_path)
    run_prediction(parsed_data)

if __name__ == "__main__":
    main()
