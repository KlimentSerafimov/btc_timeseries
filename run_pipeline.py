import os
import subprocess
import time

def run_script(script_name):
    """Run a Python script and print its output"""
    print(f"\n{'='*50}")
    print(f"Running {script_name}...")
    print(f"{'='*50}\n")
    
    process = subprocess.Popen(['python', script_name], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    # Wait for the process to complete
    process.wait()
    
    # Check if there were any errors
    if process.returncode != 0:
        print(f"\nError running {script_name}:")
        for line in process.stderr:
            print(line, end='')
    else:
        print(f"\n{script_name} completed successfully!")

def main():
    """Run the entire Bitcoin analysis pipeline"""
    start_time = time.time()
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Download data
    run_script('btc_data_downloader.py')
    
    # Step 2: Analyze data
    run_script('btc_analysis.py')
    
    # Step 3: Run modeling (commented out for now)
    # run_script('btc_modeling.py')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n{'='*50}")
    print(f"Pipeline completed in {elapsed_time:.2f} seconds!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 