import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(title="Image Annotator")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/data", StaticFiles(directory="data"), name="data")

templates = Jinja2Templates(directory="templates")

IMAGE_DIR = "data"
# ensure IMAGE_DIR exists
os.makedirs(IMAGE_DIR, exist_ok=True)

class RenameRequest(BaseModel):
    old_name: str
    new_name: str

def get_image_files():
    valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.webp')
    return [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/images")
async def list_images():
    images = get_image_files()
    # Provide the list of objects containing name and url
    return [{"name": img, "url": f"/data/{img}"} for img in images]

@app.post("/api/rename")
async def rename_image(req: RenameRequest):
    old_path = os.path.join(IMAGE_DIR, req.old_name)
    
    if not os.path.exists(old_path):
        return JSONResponse(status_code=404, content={"message": "Original image not found"})
        
    # Get extension
    _, ext = os.path.splitext(req.old_name)
    
    # Secure the new name
    safe_new_name = "".join([c for c in req.new_name if c.isalpha() or c.isdigit() or c in (' ', '-', '_')]).strip()
    if not safe_new_name:
        return JSONResponse(status_code=400, content={"message": "Invalid new name"})
        
    new_filename = f"{safe_new_name}{ext}"
    new_path = os.path.join(IMAGE_DIR, new_filename)
    
    if os.path.exists(new_path):
        return JSONResponse(status_code=400, content={"message": "An image with this name already exists"})
        
    try:
        os.rename(old_path, new_path)
        return {"message": "Image renamed successfully", "new_name": new_filename, "url": f"/data/{new_filename}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
