import kagglehub

# Download latest version
path = kagglehub.dataset_download("obulisainaren/retinal-oct-c8")

print("Path to dataset files:", path)