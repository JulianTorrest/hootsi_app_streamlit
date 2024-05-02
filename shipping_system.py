import streamlit as st
import pandas as pd

# Catálogo de Electrodomésticos
catalog_data = {
    'Product_ID': [1, 2, 3, 4, 5],
    'Product_Name': ['Refrigerator', 'Microwave', 'Dishwasher', 'Washing Machine', 'Dryer'],
    'Manufacturer': ['Samsung', 'LG', 'Whirlpool', 'GE Appliances', 'Maytag'],
    'Price': [799, 149, 499, 649, 399],
    'Description': ['Side-by-Side Refrigerator', 'Countertop Microwave', 'Built-in Dishwasher',
                    'Front-Load Washing Machine', 'Electric Dryer']
}

catalog = pd.DataFrame(catalog_data)

# Sedes del Almacén
warehouse_data = {
    'Warehouse_ID': [1, 2, 3],
    'City': ['New York', 'Los Angeles', 'Chicago'],
    'State': ['NY', 'CA', 'IL'],
    'Address': ['123 Main St', '456 Elm St', '789 Oak St']
}

warehouses = pd.DataFrame(warehouse_data)

def main():
    st.title('Warehouse Management System')

    # Sidebar
    st.sidebar.header('Select Warehouse')
    selected_warehouse = st.sidebar.selectbox('Select Warehouse', warehouses['City'])

    # Display selected warehouse address
    warehouse_address = warehouses[warehouses['City'] == selected_warehouse]['Address'].iloc[0]
    st.sidebar.write('Warehouse Address:', warehouse_address)

    # Display catalog
    st.subheader('Catalog')
    st.dataframe(catalog)

    # Product selection
    selected_product_id = st.selectbox('Select Product ID', catalog['Product_ID'])
    selected_product = catalog[catalog['Product_ID'] == selected_product_id].iloc[0]

    # Display selected product details
    st.write('Selected Product Details:')
    st.write('Name:', selected_product['Product_Name'])
    st.write('Manufacturer:', selected_product['Manufacturer'])
    st.write('Price:', selected_product['Price'])
    st.write('Description:', selected_product['Description'])

    # Order form
    st.subheader('Order Form')
    quantity = st.number_input('Enter Quantity', min_value=1, value=1)
    st.write('Total Price:', quantity * selected_product['Price'])

    # Submit button
    if st.button('Submit Order'):
        st.success(f'Order for {quantity} units of {selected_product["Product_Name"]} placed successfully!')

if __name__ == '__main__':
    main()

