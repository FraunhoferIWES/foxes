


Installation
===========================

    

Install foxes (for users)
-------------------------

* Install from PyPi.org (official releases)::
  
    pip install foxes

* Install from gitlab  (includes any recent updates)::
  
    pip install git+https://github.com/FraunhoferIWES/foxes.git
        


Install foxes (for developers)
------------------------------

1. We highly recommend developers install foxes into its own virtual environment 
(for details, see `Python Packaging User Guide <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment>`_).
For example, on a Linux system (note that you can replace ``~/venv/foxes`` by any other directory path)::

    python3 -m venv ~/venv/foxes
    source ~/venv/foxes/bin/activate

For Windows, see link above.

2. The commands to clone and install foxes with developer
options and dependencies are::

   git clone https://github.com/FraunhoferIWES/foxes.git
   cd foxes
   pip install -e .
   