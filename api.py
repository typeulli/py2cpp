import traceback
from pathlib import Path
from fastapi import FastAPI, APIRouter
from fastapi.responses import HTMLResponse
from py2cpp import py_2_cpp

path_here = Path(__file__).parent.resolve()
index_file = path_here / "index.html"
html_index = index_file.read_text(encoding="utf-8")

app = FastAPI()

router = APIRouter(prefix="/py2cpp")

@router.get("")
async def index():
    return HTMLResponse(content=html_index, status_code=200)

@router.post("/convert")
async def convert_code(request: dict[str, str]):
    code = request.get("code", "")
    try:
        cpp_code = py_2_cpp(code)
        return {"cpp_code": cpp_code}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

app.include_router(router)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7001)