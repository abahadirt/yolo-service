# SETUP:


### Clone the repository
```bash
git clone https://github.com/abahadirt/yolo-service.git
cd yolo-service
```

## Build, Run Docker:


#### Using Docker, there's no need to manually create a virtual environment or install dependencies.
#### The Dockerfile handles everything, including installing packages from requirements.txt.
```bash
docker build -t yolo-onnx-service .
docker run -d -p 8000:8000 --name yolo_app yolo-onnx-service  
docker logs yolo_app
```
to test endpoints: http://localhost:8000/docs

## Development setup in optimize-docker branch:

#### The virtual environment folder used in this branch is named venv_optimized.
#### (Use the default virtual environment name venv in the development branch.)
```bash
python -m venv venv_optimized
# For Windows:
venv_optimized\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app
```
## ..


##  Project description is available in the `development` branch.
