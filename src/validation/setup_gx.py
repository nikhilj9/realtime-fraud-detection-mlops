"""
Initialize Great Expectations File Data Context.

GX Core 1.10.0 uses Python instead of CLI.
Running this script creates the gx/ folder structure.
"""

import great_expectations as gx


def init_gx_context() -> None:
    """Create GX file context if it doesn't exist."""
    # mode="file" creates gx/ folder with config files
    # If already exists, it loads the existing context
    context = gx.get_context(mode="file")
    print(f"GX Context ready at: {context.root_directory}")
    print("Great Expectations initialized successfully!")


if __name__ == "__main__":
    init_gx_context()
