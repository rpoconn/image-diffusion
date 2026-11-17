import os
import tempfile

# Settings are read at import time, so set them before the app is imported.
os.environ["RENDERER"] = "mock"
os.environ["OUTPUT_DIR"] = tempfile.mkdtemp(prefix="clothing-designer-test-")
