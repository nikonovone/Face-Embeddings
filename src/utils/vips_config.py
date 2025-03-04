import ctypes
import os


def configure_vips():
    """Configure VIPS library paths"""
    # Add library paths
    os.environ["LD_LIBRARY_PATH"] = "/usr/lib64:/usr/local/lib:" + os.environ.get(
        "LD_LIBRARY_PATH",
        "",
    )

    try:
        # Try to load the library directly
        if not any(map(lambda x: "libvips.so.42" in x, os.listdir("/usr/lib64"))):
            raise OSError("libvips.so.42 not found in /usr/lib64")

        ctypes.CDLL("/usr/lib64/libvips.so.42")
    except Exception as e:
        print(f"Warning: Failed to load libvips directly: {e}")
        return False

    return True
