print('print from python')

import os
from dotenv import load_dotenv

# Load env
load_dotenv()

print('DATA_DIR={}'.format(os.environ['DATA_DIR']))

print('end of python file')