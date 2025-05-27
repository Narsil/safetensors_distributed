import numpy as np
import unittest
import subprocess
import time
import os
import signal
import atexit
import tempfile
import shutil
from safetensors.numpy import save_file
from safetensors_distributed import dist_loader, SafetensorDistributedError


class TestPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for our test files
        cls.test_dir = tempfile.mkdtemp()

        # Create a dummy tensor and save it as safetensors

        # Start the HTTP server in the test directory
        cmd = ["python", "-m", "http.server", "8000"]
        cls.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cls.test_dir,  # Run server from test directory
        )
        # Wait a moment for the server to start
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        # Cleanup: terminate the server process
        if cls.server_process:
            cls.server_process.terminate()
            cls.server_process.wait()

        # Clean up the temporary directory
        shutil.rmtree(cls.test_dir)

    def test_plan_execute_simple(self):
        tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        self.test_file_path = os.path.join(self.test_dir, "test.safetensors")
        save_file({"a": tensor}, self.test_file_path)
        url = "http://127.0.0.1:8000/test.safetensors"
        with dist_loader(url) as loader:
            plan = loader.plan()
            # Use tensor 'a' with shape (2, 2)
            plan.add_slice(loader.get_slice("a")[:2, :1])
            result = plan.execute(loader)

        self.assertIn("a", result)
        arr = result["a"]
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (2, 1))

        # Now we can check the actual values since we know what we put in
        expected = np.array(
            [
                [
                    1.0,
                ],
                [3.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(arr, expected)
        np.testing.assert_allclose(arr, tensor[:2, :1])

    def test_plan_invalid_double(self):
        tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        self.test_file_path = os.path.join(self.test_dir, "test.safetensors")
        save_file({"a": tensor}, self.test_file_path)
        url = "http://127.0.0.1:8000/test.safetensors"
        with dist_loader(url) as loader:
            plan = loader.plan()
            # Use tensor 'a' with shape (2, 2)
            plan.add_slice(loader.get_slice("a")[:2, :1])
            with self.assertRaises(SafetensorDistributedError):
                plan.add_slice(loader.get_slice("a")[:1, :2])

    def test_plan_execute_partial(self):
        tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        self.test_file_path = os.path.join(self.test_dir, "test.safetensors")
        save_file({"a": tensor}, self.test_file_path)
        url = "http://127.0.0.1:8000/test.safetensors"
        with dist_loader(url) as loader:
            plan = loader.plan()
            # Use tensor 'a' with shape (2, 2)
            plan.add_slice(loader.get_slice("a")[:1])
            result = plan.execute(loader)

        self.assertIn("a", result)
        arr = result["a"]
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (1, 2))

        # Now we can check the actual values since we know what we put in
        expected = np.array(
            [
                [1.0, 2.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(arr, expected)
        np.testing.assert_allclose(arr, tensor[:1])


if __name__ == "__main__":
    unittest.main()
