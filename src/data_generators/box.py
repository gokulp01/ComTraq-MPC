import math

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth's radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi/2)**2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c

# Coordinates
sw = [25.7581072, -80.3738942]
se = [25.7581072, -80.3734494]
nw = [25.7583659, -80.3738942]
ne = [25.7583659, -80.3734494]

# Calculate width
width = haversine_distance(sw[0], sw[1], se[0], se[1])

# Calculate height
height = haversine_distance(sw[0], sw[1], nw[0], nw[1])

print(f"Width of the bounding box: {width:.2f} meters")
print(f"Height of the bounding box: {height:.2f} meters")
