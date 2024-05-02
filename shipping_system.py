import streamlit as st

# Define the initial inventory (product name, quantity)
inventory = {
    "Product A": 10,
    "Product B": 15,
    "Product C": 20
}

# Function to deduct products from inventory
def deduct_inventory(product, quantity):
    if product in inventory:
        if inventory[product] >= quantity:
            inventory[product] -= quantity
            return True
        else:
            st.error(f"Not enough {product} quantity in inventory.")
            return False
    else:
        st.error("The selected product is not in inventory.")
        return False

# User interface with Streamlit
def main():
    st.title("Shipping System")

    # Show the current inventory
    st.subheader("Current Inventory")
    st.write(inventory)

    # Select product and quantity to ship
    st.subheader("Select Product to Ship")
    product = st.selectbox("Select a product", list(inventory.keys()))
    quantity = st.number_input("Quantity to ship", min_value=1, max_value=inventory[product])

    # Button to ship products
    if st.button("Ship Product"):
        if deduct_inventory(product, quantity):
            st.success(f"{quantity} units of {product} shipped successfully.")
            # You can add logic here to log shipping details

if __name__ == "__main__":
    main()
