import unittest
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", help="The start directory", default=".")
    parser.add_argument(
        "-p", "--pattern", help="The file search pattern", default="test*.py"
    )
    parser.add_argument("-v", "--verbosity", help="The verbosity", type=int, default=2)
    args = parser.parse_args()

    testsuite = unittest.TestLoader().discover(args.root, pattern=args.pattern)
    unittest.TextTestRunner(verbosity=args.verbosity).run(testsuite)
