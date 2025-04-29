from kedro.io import DataCatalog
from kedro.config import AbstractConfigLoader as ConfigLoader
from pathlib import Path

# Load configuration
conf_paths = ["conf/base", "conf/local"]
config_loader = ConfigLoader(conf_paths)
catalog_config = config_loader.get("catalog*")

# Instantiate catalog
try:
    catalog = DataCatalog.from_config(catalog_config)
    print("Catalog loaded successfully!")
    print(catalog.list())
except Exception as e:
    print(f"Error loading catalog: {e}")
    raise