import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2) * math.sin(dlat/2) +
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
        math.sin(dlon/2) * math.sin(dlon/2))
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)) * 1000)  # Convert to meters

def new_coords(lat, lon, distance_meters, direction_deg):
    """
    Calculate new geographic coordinates after moving a specified distance in a given direction.
    
    Parameters:
        lat (float): Initial latitude in degrees
        lon (float): Initial longitude in degrees
        distance_meters (float): Distance to move (5-20 meters)
        direction_deg (float): Compass direction (0-360 degrees)
    
    Returns:
        tuple: (new_lat, new_lon) in degrees
    """
    # Earth's radius in meters
    R = 6371e3  # Mean Earth radius
    
    # Convert degrees to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing = math.radians(direction_deg)
    
    # Calculate displacement components
    delta_lat = (distance_meters * math.cos(bearing)) / R
    delta_lon = (distance_meters * math.sin(bearing)) / (R * math.cos(lat_rad))
    
    # Calculate new coordinates in radians
    new_lat_rad = lat_rad + delta_lat
    new_lon_rad = lon_rad + delta_lon
    
    # Convert back to degrees
    new_lat = math.degrees(new_lat_rad)
    new_lon = math.degrees(new_lon_rad)
    
    return (round(new_lat, 6), round(new_lon, 6))  # ~0.1m precision