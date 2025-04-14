
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import json
import uuid
import asyncio
import subprocess
import os
import time

# --------------------------------------------------------------------------
# Import ComfyUI's components (adjust paths as needed)
# You might need to figure out how to import these correctly
# or if you need to use subprocesses instead
# Example:
# from comfy.model_management import load_model_from_file
# from comfy.workflow import load_workflow_json, execute_workflow
# --------------------------------------------------------------------------

app = FastAPI()

# Data Models (for request validation and data structuring)
class ImageGenerationRequest(BaseModel):
    workflow_path: str
    prompt: str


class ModelTrainingRequest(BaseModel):
    model_name: str
    dataset_path: str
    epochs: int

class ModelLoadRequest(BaseModel):
    model_path: str

class TaskStatus(BaseModel):
    task_id: str
    status: str
    message: str = ""
    progress: float = 0.0


# In-memory task tracking (replace with a database for production)
tasks = {}

# ComfyUI Configuration (Adjust as needed)
COMFYUI_PATH = "./../../"  # Replace with your ComfyUI path
COMFYUI_WORKFLOW_FOLDER = os.path.join(COMFYUI_PATH, "workflows")
COMFYUI_OUTPUT_FOLDER = os.path.join(COMFYUI_PATH, "output")
COMFYUI_SCRIPT = os.path.join(COMFYUI_PATH, "main.py")  # Or a custom script

def execute_comfyui_workflow(workflow_path, prompt, task_id):
    """
    Executes a ComfyUI workflow using a subprocess.
    """
    try:
        # Example using subprocess - you might need to adapt this
        command = [
            "python",
            COMFYUI_SCRIPT,
            "--workflow-path",
            workflow_path,
            "--prompt",
            prompt,
            "--output-folder",
            COMFYUI_OUTPUT_FOLDER,
            "--task-id",
            task_id,
        ]
        print(f"Executing ComfyUI with command: {' '.join(command)}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=COMFYUI_PATH)

        stdout, stderr = process.communicate()
        print(f"ComfyUI Output: {stdout.decode()}")
        if stderr:
            print(f"ComfyUI Error: {stderr.decode()}")
        
        # Simulate progress update, you need to adjust this according to the output of your subprocess.
        tasks[task_id]["progress"] = 0.5
        time.sleep(5)
        tasks[task_id]["progress"] = 1.0

        # Check for errors, etc.
        if process.returncode != 0:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["message"] = f"ComfyUI execution failed: {stderr.decode()}"
            return None
        
        # Find the generated image
        for filename in os.listdir(COMFYUI_OUTPUT_FOLDER):
            if filename.startswith(f"ComfyUI_{task_id}_output_"):
                result_image_path = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
                tasks[task_id]["status"] = "completed"
                tasks[task_id]["message"] = f"Image saved at {result_image_path}"
                return result_image_path
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = "Image not generated"
        return None
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = str(e)
        return None

# Image Generation Endpoint
@app.post("/generate_image/", response_model=TaskStatus)
async def generate_image_endpoint(request: ImageGenerationRequest):
    """
    Generates an image based on the provided workflow and prompt.
    """
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending"}
    
    workflow_path = os.path.join(COMFYUI_WORKFLOW_FOLDER, request.workflow_path)

    async def image_generation_task():
        try:
            tasks[task_id]["status"] = "running"
            result_image_path = execute_comfyui_workflow(workflow_path, request.prompt, task_id)
            if result_image_path:
                print(result_image_path)
            # Simulate a long-running task:
            await asyncio.sleep(2) #wait 2 seconds to finish the task
        except Exception as e:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["message"] = str(e)

    asyncio.create_task(image_generation_task())

    return JSONResponse(content={"task_id": task_id, "status": "pending"}, status_code=202)

# Model Training Endpoint (Illustrative)
@app.post("/train_model/", response_model=TaskStatus)
async def train_model_endpoint(request: ModelTrainingRequest):
    """
    Starts training a model with the specified parameters.
    """
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending"}

    async def model_training_task():
        try:
            tasks[task_id]["status"] = "running"
            model_name = request.model_name
            dataset_path = request.dataset_path
            epochs = request.epochs

            print(f"Training model '{model_name}' with dataset '{dataset_path}' for {epochs} epochs")

            # Placeholder: Adapt this to the real ComfyUI training command
            # (e.g., using subprocess or a ComfyUI API if it exists)
            # subprocess.run(["python", "train_comfyui.py", ...], check=True)

            # Simulating a long-running task:
            await asyncio.sleep(5)
            tasks[task_id]["progress"] = 0.25
            await asyncio.sleep(10)
            tasks[task_id]["progress"] = 0.5
            await asyncio.sleep(10)
            tasks[task_id]["progress"] = 0.75
            await asyncio.sleep(5)
            tasks[task_id]["progress"] = 1.0

            tasks[task_id]["status"] = "completed"
            tasks[task_id]["message"] = "Model training finished successfully."
        except Exception as e:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["message"] = str(e)

    asyncio.create_task(model_training_task())

    return JSONResponse(content={"task_id": task_id, "status": "pending"}, status_code=202)

# Model Loading Endpoint (Illustrative)
@app.post("/load_model/", response_model=TaskStatus)
async def load_model_endpoint(request: ModelLoadRequest):
    """
    Loads a model into ComfyUI.
    """
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending"}

    async def model_loading_task():
        try:
            tasks[task_id]["status"] = "running"
            model_path = request.model_path

            print(f"Loading model from '{model_path}'")

            # Placeholder: Adapt this to the real ComfyUI model loading mechanism
            # (e.g., using subprocess or a ComfyUI API if it exists)
            # subprocess.run(["python", "load_model_comfyui.py", ...], check=True)
            # or
            # load_model_from_file(model_path)

            # Simulating a long-running task:
            await asyncio.sleep(2)
            tasks[task_id]["progress"] = 1.0

            tasks[task_id]["status"] = "completed"
            tasks[task_id]["message"] = f"Model '{model_path}' loaded successfully."
        except Exception as e:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["message"] = str(e)

    asyncio.create_task(model_loading_task())

    return JSONResponse(content={"task_id": task_id, "status": "pending"}, status_code=202)

@app.get("/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """
    Gets the status of a specific task (image generation, training, etc.).
    """
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.get("/get_image/{task_id}")
async def get_image(task_id: str):
    """
    Gets the image result of a specific task.
    """
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")
    if not os.path.exists(task["message"].replace("Image saved at ", "")):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(task["message"].replace("Image saved at ", ""))
