import asyncio
import json
import subprocess
import threading
import time
import traceback
from typing import Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from py2cpp import py_2_cpp
from py2cpp.core.compiler import Setting, NameDict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/env/py2cpp_header.py")
async def get_header():
    return FileResponse("py2cpp_header.py")

_DATA_NAMES = {"namespaces": list(NameDict.keys())}
@app.get("/env/namespaces")
async def get_env_names():
    return _DATA_NAMES

class ConvertRequest(BaseModel):
    code: str
    filename: str = "<string>"
    namespaces: list[str] = []

def pad_comment(text: str):
    lines = text.splitlines()
    if not lines:
        return "//"
    padded_lines = ["// " + lines[0]]
    for line in lines[1:]:
        padded_lines.append("// " + line)
    return "\n".join(padded_lines)

@app.post("/convert")
async def convert_code(request: ConvertRequest):
    try:
        setting = Setting(minimize_namespace=request.namespaces)
        cpp_code = py_2_cpp(request.code, request.filename, setting=setting).code
        return {"state": "success", "result": cpp_code}
    except SyntaxError as e:
        message = "".join(traceback.format_exception_only(SyntaxError, e)).strip()
        return {"state": "error", "message": pad_comment(message)}
    except Exception as e:
        traceback.print_exc()
        return {"state": "error", "message": pad_comment(f"{e.__class__.__name__}: {e!s}")}


class PyrightSession:
    def __init__(self, uri: str):
        self.uri = uri
        self.version = 1
        self.responses: dict[int, Any] = {}
        self.proc = subprocess.Popen(
            ["pyright-langserver", "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        threading.Thread(target=self._read_output, daemon=True).start()
        self._initialize()

    def _send(self, req: dict[str, Any]):
        assert self.proc.stdin is not None
        data = json.dumps(req)
        msg = f"Content-Length: {len(data)}\r\n\r\n{data}"
        self.proc.stdin.write(msg.encode())
        self.proc.stdin.flush()

    def _read_output(self):
        assert self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            if line.startswith(b"Content-Length"):
                length = int(line.split(b":")[1].strip())
                self.proc.stdout.readline()
                body = self.proc.stdout.read(length)
                msg = json.loads(body)
                if "id" in msg:
                    self.responses[msg["id"]] = msg

    def _initialize(self):
        self._send({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"processId": None, "rootUri": None, "capabilities": {}}
        })
        # wait for response
        for _ in range(50):
            if 1 in self.responses:
                _ = self.responses.pop(1)
                break
            time.sleep(0.01)
        # initialized notification
        self._send({"jsonrpc": "2.0", "method": "initialized", "params": {}})

    def update_code(self, code: str):
        # first update is didOpen, subsequent updates are didChange
        if self.version == 1:
            self._send({
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": self.uri,
                        "languageId": "python",
                        "version": self.version,
                        "text": code
                    }
                }
            })
        else:
            self._send({
                "jsonrpc": "2.0",
                "method": "textDocument/didChange",
                "params": {
                    "textDocument": {"uri": self.uri, "version": self.version},
                    "contentChanges": [{"text": code}]
                }
            })
        self.version += 1

    async def get_completion(self, line: int, character: int):
        req_id = int(time.time() * 1000)
        self._send({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "textDocument/completion",
            "params": {
                "textDocument": {"uri": self.uri},
                "position": {"line": line, "character": character}
            }
        })
        for _ in range(50):
            if req_id in self.responses:
                return self.responses.pop(req_id)
            await asyncio.sleep(0.01)
        return {"error": "timeout"}


sessions: dict[str, PyrightSession] = {}

@app.websocket("/ws/{doc_id}")
async def websocket_endpoint(ws: WebSocket, doc_id: str):
    await ws.accept()
    uri = f"file:///{doc_id}.py"
    session = PyrightSession(uri)
    sessions[doc_id] = session

    try:
        while True:
            data = await ws.receive_json()
            if data["type"] == "update":
                session.update_code(data["code"])
                await ws.send_json({"status": "updated", "version": session.version})
            elif data["type"] == "complete":
                res = await session.get_completion(data["line"], data["character"])
                await ws.send_json(res)
    except WebSocketDisconnect: pass
    except Exception as e:      print(f"Unexpected error: {e}")
    finally:
        session.proc.terminate()
        sessions.pop(doc_id, None)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=7001, reload=True)