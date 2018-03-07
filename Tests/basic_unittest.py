import unittest
import basic_optimization

'''
This is a dummy basic file setup for unittests. 
Will expand on it soon enough
'''


class TestBasicOptimization(unittest.TestCase):

    def test_basic(self):

        basic = basic_optimization.BasicOptimization(function_type='diff', method_type='grad', is_feature=True, is_stoc=False)
        x,g, f = basic.run(step_size=1.)
        self.assertEqual(x, -9.)

if __name__ == '__main__':
    unittest.main()