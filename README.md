# This README is under construction

# Installation

1. Clone this repository

'''
git clone https://github.com/barcarolo01/bearingEstimation.git
'''

2. Create a python virtual environment

'''
python -m venv .venv
'''

3. Install hydromate (python version) as a package
Note: Little modification to hydromate are required to successfully install it. This will be fixed shortly.

'''
pip install -e path/to/hydromate/folder
'''

Please refers to the official page of the project for more details.

4. Install digitalshadow as a package
'''
pip install -e path/to/digitalshadow/folder
'''

5. Create a `.env` file with the following environment variables

```env
SAMPLING_FREQUENCY = "96000"
NUMBER_OF_HYDROPHONES = "5"
HYDROMATE_PY_PATH = "Path/to/your/hydromate/python/hm_code/folder"
```