https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/ttm_getting_started.ipynb



This modified script introduces caching at several levels:

Data caching: The get_cached_data function caches the dataset splits, so if you run the script multiple times with the same parameters, it won't need to reload the data from disk each time.
Zero-shot evaluation caching: The results of the zero-shot evaluation are cached, so if you run the same evaluation again, it will use the cached results instead of re-evaluating.
Few-shot fine-tuning evaluation caching: Similarly, the results of the few-shot fine-tuning are cached based on the parameters used.

The cache uses the EnhancedJSONCache class we created earlier, which stores the data in a JSON file (ttm_cache.json). The cache has an expiry time of 24 hours (86400 seconds) and a maximum size of 1000 entries.
To use this script:

Make sure the enhanced_json_cache.py file (containing the EnhancedJSONCache class) is in the same directory as this script.
Run the script as before.

The first time you run the script, it will perform all computations as normal. On subsequent runs with the same parameters, it will use the cached results where available, potentially saving a significant amount of time.
Note that this caching system is storing the entire datasets and model outputs in memory and writing them to a JSON file. For very large datasets or outputs, this might not be the most efficient approach, and you might want to consider more sophisticated caching mechanisms or databases for storing intermediate results.


