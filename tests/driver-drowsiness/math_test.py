import unittest
# import our `pybind11`-based extension module from package python_cpp_example
#from python_cpp_example import python_cpp_example

class MainTest(unittest.TestCase):
    def test_add(self):
        # test that 1 + 1 = 2
        self.assertEqual(1 + 1, 2)

    def test_subtract(self):
        # test that 1 - 1 = 0
        self.assertEqual(1 - 1, 0)

if __name__ == '__main__':
    unittest.main()
	