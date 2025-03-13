import os, sys
import requests
from pathlib import Path

# Add project root to Python path
# project_root = Path(__file__).resolve().parent.parent.parent
# sys.path.append(str(project_root))

from generic.config import API_KEY

class StreetViewAPI:
    def __init__(self, save_folder=None):
        self.base_url = "https://maps.googleapis.com/maps/api/streetview"
        self.metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
        # Resolve save folder path
        if save_folder is None:
            # Default to project_root/generic/images
            self.save_folder = Path(__file__).resolve().parent.parent.parent / "generic" / "images"
        else:
            self.save_folder = Path(save_folder).resolve()
        os.makedirs(self.save_folder, exist_ok=True)

    def _geocode(self, address):
        geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
        response = requests.get(geocode_url, params={"address": address, "key": API_KEY})
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            raise ValueError(f"No coordinates found for address: {address}")
        return results[0]["geometry"]["location"]

    def download_image(self, lat=None, lon=None, address=None, heading=0, pitch=0, fov=120, size="600x400"):
        if address:
            location = self._geocode(address)
            lat, lon = location["lat"], location["lng"]
        
        params = {
            "size": size,
            "location": f"{lat},{lon}",
            "heading": heading,
            "pitch": pitch,
            "fov": fov,
            "key": API_KEY
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        
        filename = f"sv_{lat:.6f}_{lon:.6f}_{params.get('heading',0)}.jpg"
        filepath = os.path.join(self.save_folder, filename)
        
        with open(filepath, "wb") as f:
            f.write(response.content)
            
        return filepath

    def get_metadata(self, lat, lon):
        response = requests.get(self.metadata_url, params={
            "location": f"{lat},{lon}",
            "key": API_KEY
        })
        return response.json()