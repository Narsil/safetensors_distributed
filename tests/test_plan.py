import numpy as np
import pytest
import subprocess
import time
import os
import signal
import atexit
from safetensors_distributed import dist_loader, PyPlan

# Global variable to hold the subprocess
server_process = None

@pytest.fixture(scope="module")
def http_server():
    global server_process
    # Use Python's http.server module in a subprocess on a fixed port (8000)
    cmd = ["python", "-m", "http.server", "8000"]
    server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Wait a moment for the server to start
    time.sleep(0.5)
    yield 8000
    # Cleanup: terminate the server process
    if server_process:
        server_process.terminate()
        server_process.wait()

def test_plan_execute(http_server):
    url = f"http://127.0.0.1:{http_server}/test.safetensors"
    loader = dist_loader(url)
    plan = PyPlan()
    # Use tensor 'a' with shape (2, 2)
    plan.get_slice(loader, "a", slice(0, 2))
    result = plan.execute(loader)
    assert "a" in result
    arr = result["a"]
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)
    # Optionally check values if known
    # expected = np.array([[...], [...]])
    # np.testing.assert_allclose(arr, expected) 