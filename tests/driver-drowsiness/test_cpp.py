import unittest
import subprocess
import os

class MainTest(unittest.TestCase):
    def test_cpp(self):
        print("\n\nTesting C++ code...")
        path = os.path.join(os.path.dirname(__file__), '..', '..', 'bin', 'Debug', 'driver-drowsiness_test.exe')
        print(path)
        subprocess.check_call(path)


if __name__ == '__main__':
    unittest.main()
