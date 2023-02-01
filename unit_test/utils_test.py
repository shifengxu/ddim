import unittest
import utils


class UtilsTest(unittest.TestCase):

    def test_create_geometric_series(self):
        res = utils.create_geometric_series(1, 8., 2., 4)  # ratio = 2
        print(res)
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0], 1.)
        self.assertEqual(res[1], 2.)
        self.assertEqual(res[2], 4.)
        self.assertEqual(res[3], 8.)
        res = utils.create_geometric_series(0, 8., 1., 9)  # ratio = 1
        print(res)
        self.assertEqual(len(res), 9)
        self.assertEqual(res[0], 0.)
        self.assertEqual(res[1], 1.)
        self.assertEqual(res[2], 2.)
        self.assertEqual(res[8], 8.)
        res = utils.create_geometric_series(1, 8., 0.5, 4)  # ratio = 0.5
        print(res)
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0], 1.)
        self.assertEqual(res[1], 5.)
        self.assertEqual(res[2], 7.)
        self.assertEqual(res[3], 8.)
        res = utils.create_geometric_series(8, 1., 2., 4)  # ratio = 2
        print(res)
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0], 8.)
        self.assertEqual(res[1], 7.)
        self.assertEqual(res[2], 5.)
        self.assertEqual(res[3], 1.)

if __name__ == '__main__':
    unittest.main()