import sys
import os
import pkg_resources

class EnvironmentSnapshot:
    """
    Captures a snapshot of the Python environment.
    Includes installed packages and their versions.
    """
    def capture(self):
        installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        return {
            "python_version": sys.version,
            "packages": installed_packages,
            "cwd": os.getcwd()
        }
