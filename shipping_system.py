import streamlit as st

# Catálogo de productos ampliado
product_catalog = [
    {"name": "Refrigerator", "price": 799, "description": "Large capacity refrigerator."},
    {"name": "Washing Machine", "price": 499, "description": "Front-loading washing machine."},
    {"name": "Dishwasher", "price": 349, "description": "Energy-efficient dishwasher."},
    {"name": "Microwave Oven", "price": 149, "description": "Compact microwave oven."},
    {"name": "Toaster", "price": 29, "description": "Stainless steel toaster."},
    {"name": "Blender", "price": 69, "description": "High-speed blender."},
    {"name": "Coffee Maker", "price": 99, "description": "Programmable coffee maker."},
    {"name": "Food Processor", "price": 129, "description": "Multifunctional food processor."},
    {"name": "Stand Mixer", "price": 249, "description": "Professional stand mixer."},
    {"name": "Vacuum Cleaner", "price": 199, "description": "Bagless upright vacuum cleaner."},
    {"name": "Air Purifier", "price": 299, "description": "HEPA air purifier."},
    {"name": "Smart Thermostat", "price": 179, "description": "Wi-Fi enabled smart thermostat."},
    {"name": "Smart Speaker", "price": 99, "description": "Voice-controlled smart speaker."},
    {"name": "Smart TV", "price": 699, "description": "4K Ultra HD smart TV."},
    {"name": "Laptop", "price": 999, "description": "Thin and light laptop."},
    {"name": "Tablet", "price": 399, "description": "High-resolution tablet."},
    {"name": "Smartphone", "price": 799, "description": "5G smartphone with OLED display."},
    {"name": "Digital Camera", "price": 599, "description": "Mirrorless digital camera."},
    {"name": "Drone", "price": 999, "description": "Professional aerial drone."},
    {"name": "Gaming Console", "price": 499, "description": "Next-gen gaming console."},
    {"name": "Smart Watch", "price": 349, "description": "Fitness tracking smart watch."},
    {"name": "Electric Scooter", "price": 399, "description": "Foldable electric scooter."},
    {"name": "Robot Vacuum", "price": 299, "description": "Self-charging robot vacuum."},
    {"name": "Portable Speaker", "price": 79, "description": "Waterproof portable speaker."},
    {"name": "Wireless Headphones", "price": 199, "description": "Noise-cancelling wireless headphones."},
    {"name": "Home Security Camera", "price": 149, "description": "Indoor/outdoor home security camera."},
    {"name": "Wireless Router", "price": 129, "description": "High-speed wireless router."},
    {"name": "External Hard Drive", "price": 79, "description": "Portable external hard drive."},
    {"name": "Digital Photo Frame", "price": 59, "description": "LCD digital photo frame."},
    {"name": "Smart Doorbell", "price": 199, "description": "Video doorbell with motion detection."},
    {"name": "Power Bank", "price": 49, "description": "Portable power bank charger."},
    {"name": "Fitness Tracker", "price": 129, "description": "Activity and sleep tracking fitness tracker."},
    {"name": "Wireless Mouse", "price": 29, "description": "Ergonomic wireless mouse."},
    {"name": "LED Desk Lamp", "price": 39, "description": "Adjustable LED desk lamp."},
    {"name": "Digital Voice Recorder", "price": 59, "description": "Voice-activated digital voice recorder."},
    {"name": "Bluetooth Keyboard", "price": 49, "description": "Compact Bluetooth keyboard."},
    {"name": "USB-C Hub", "price": 29, "description": "Multiport USB-C hub adapter."},
    {"name": "Wireless Charging Pad", "price": 39, "description": "Qi-certified wireless charging pad."},
    {"name": "Car Dash Cam", "price": 99, "description": "1080p HD car dash cam."},
    {"name": "Mini Projector", "price": 149, "description": "Portable mini LED projector."},
    {"name": "Electric Toothbrush", "price": 79, "description": "Sonic electric toothbrush with UV sanitizer."},
    {"name": "Air Fryer", "price": 119, "description": "Oil-less air fryer for healthier cooking."},
    {"name": "Smart Lock", "price": 149, "description": "Keyless smart door lock with touchscreen."},
    {"name": "Wireless Earbuds", "price": 99, "description": "True wireless earbuds with charging case."},
    {"name": "Induction Cooktop", "price": 199, "description": "Portable induction cooktop for precise cooking."},
    {"name": "Fitness Smart Scale", "price": 69, "description": "Bluetooth-enabled fitness smart scale."},
    {"name": "UV Sterilizer Box", "price": 59, "description": "UV sterilizer box for disinfecting items."},
    {"name": "Electric Kettle", "price": 49, "description": "Stainless steel electric kettle with temperature control."},
    {"name": "Smart Plug", "price": 29, "description": "Wi-Fi smart plug with energy monitoring."},
    {"name": "Electric Fireplace", "price": 299, "description": "Freestanding electric fireplace with remote control."},
    {"name": "Massage Gun", "price": 129, "description": "Deep tissue percussion massage gun."},
    {"name": "Smart Ceiling Fan", "price": 199, "description": "Wi-Fi enabled smart ceiling fan with light."},
    {"name": "Smart Scale", "price": 79, "description": "Bluetooth smart scale with body composition analysis."},
    {"name": "Smart Blender", "price": 149, "description": "App-controlled smart blender with pre-programmed settings."},
]

# Ubicaciones del almacén ampliadas
warehouse_locations = {
    "New York": {"address": "123 Main St, New York, NY", "inventory": {...}},
    "Los Angeles": {"address": "456 Elm St, Los Angeles, CA", "inventory": {...}},
    "Chicago": {"address": "789 Oak St, Chicago, IL", "inventory": {...}},
    "Houston": {"address": "101 Pine St, Houston, TX", "inventory": {...}},
    "Phoenix": {"address": "234 Maple St, Phoenix, AZ", "inventory": {...}},
    "Philadelphia": {"address": "567 Cedar St, Philadelphia, PA", "inventory": {...}},
    "San Antonio": {"address": "890 Birch St, San Antonio, TX", "inventory": {...}},
    "San Diego": {"address": "111 Elm St, San Diego, CA", "inventory": {...}},
    "Dallas": {"address": "222 Oak St, Dallas, TX", "inventory": {...}},
    "San Jose": {"address": "333 Maple St, San Jose, CA", "inventory": {...}},
    "Austin": {"address": "444 Cedar St, Austin, TX", "inventory": {...}},
    "Jacksonville": {"address": "555 Birch St, Jacksonville, FL", "inventory": {...}},
    "Fort Worth": {"address": "666 Elm St, Fort Worth, TX", "inventory": {...}},
    "Columbus": {"address": "777 Oak St, Columbus, OH", "inventory": {...}},
    "Charlotte": {"address": "888 Maple St, Charlotte, NC", "inventory": {...}},
    "San Francisco": {"address": "999 Cedar St, San Francisco, CA", "inventory": {...}},
    "Indianapolis": {"address": "1010 Birch St, Indianapolis, IN", "inventory": {...}},
    "Seattle": {"address": "1111 Elm St, Seattle, WA", "inventory": {...}},
    "Denver": {"address": "1212 Oak St, Denver, CO", "inventory": {...}},
    "Washington": {"address": "1313 Maple St, Washington, DC", "inventory": {...}},
    "Boston": {"address": "1414 Cedar St, Boston, MA", "inventory": {...}},
    "El Paso": {"address": "1515 Elm St, El Paso, TX", "inventory": {...}},
    "Detroit": {"address": "1616 Oak St, Detroit, MI", "inventory": {...}},
    "Nashville": {"address": "1717 Maple St, Nashville, TN", "inventory": {...}},
    "Portland": {"address": "1818 Cedar St, Portland, OR", "inventory": {...}},
    "Oklahoma City": {"address": "1919 Elm St, Oklahoma City, OK", "inventory": {...}},
    "Las Vegas": {"address": "2020 Oak St, Las Vegas, NV", "inventory": {...}},
    "Memphis": {"address": "2121 Maple St, Memphis, TN", "inventory": {...}},
    "Louisville": {"address": "2222 Cedar St, Louisville, KY", "inventory": {...}},
    "Baltimore": {"address": "2323 Elm St, Baltimore, MD", "inventory": {...}},
    "Milwaukee": {"address": "2424 Oak St, Milwaukee, WI", "inventory": {...}},
    "Albuquerque": {"address": "2525 Maple St, Albuquerque, NM", "inventory": {...}},
    "Tucson": {"address": "2626 Cedar St, Tucson, AZ", "inventory": {...}},
    "Fresno": {"address": "2727 Elm St, Fresno, CA", "inventory": {...}},
    "Sacramento": {"address": "2828 Oak St, Sacramento, CA", "inventory": {...}},
    "Mesa": {"address": "2929 Maple St, Mesa, AZ", "inventory": {...}},
    "Kansas City": {"address": "3030 Cedar St, Kansas City, MO", "inventory": {...}},
    "Atlanta": {"address": "3131 Elm St, Atlanta, GA", "inventory": {...}},
    "Long Beach": {"address": "3232 Oak St, Long Beach, CA", "inventory": {...}},
    "Colorado Springs": {"address": "3333 Maple St, Colorado Springs, CO", "inventory": {...}},
    "Raleigh": {"address": "3434 Cedar St, Raleigh, NC", "inventory": {...}},
    "Miami": {"address": "3535 Elm St, Miami, FL", "inventory": {...}},
    "Virginia Beach": {"address": "3636 Oak St, Virginia Beach, VA", "inventory": {...}},
    "Omaha": {"address": "3737 Maple St, Omaha, NE", "inventory": {...}},
    "Oakland": {"address": "3838 Cedar St, Oakland, CA", "inventory": {...}},
    "Minneapolis": {"address": "3939 Elm St, Minneapolis, MN", "inventory": {...}},
    "Tulsa": {"address": "4040 Oak St, Tulsa, OK", "inventory": {...}},
}

# Visualización de la aplicación
st.title("Logistics Application")

# Función para seleccionar un producto del catálogo
def select_product(product_catalog):
    st.subheader("Select a Product")
    product_names = [product["name"] for product in product_catalog]
    selected_product = st.selectbox("Choose a Product", product_names)
    return selected_product

# Función para seleccionar una ubicación del almacén
def select_location(warehouse_locations):
    st.subheader("Select a Warehouse Location")
    location_names = list(warehouse_locations.keys())
    selected_location = st.selectbox("Choose a Location", location_names)
    return selected_location

# Función principal
def main():
    product = select_product(product_catalog)
    location = select_location(warehouse_locations)
    st.write("You selected:", product, "located in", location)

# Ejecución de la función principal
if __name__ == "__main__":
    main()
