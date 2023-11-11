import re
import numpy as np
from tqdm import tqdm


# Step 4: Convert the dictionary to a numpy array
np_array = np.array(list(word_dict.items()))

print(np_array)
# Step 5: Save the numpy array as a .npz file
np.savez('non-neural_model/danske_ord.npz', np_array)