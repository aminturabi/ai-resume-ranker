import streamlit as st

# Function to show the "Paywall"
def show_payment_page():
    st.title("🛡️ Upgrade to AI Ranker Pro")
    st.write("Unlock advanced screening and unlimited resume rankings.")
    
    st.markdown("### Choose Your Plan")
    
    # You can create different links for different amounts in SadaBiz
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Single Report**")
        st.write("Rank up to 10 resumes")
        st.link_button("Pay $5", "https://sbiz.me/amin/5usd") # Your actual link
        
    with col2:
        st.success("**Unlimited Access**")
        st.write("Full screening for 1 month")
        st.link_button("Pay $25", "https://sbiz.me/amin/25usd") # Your actual link

    st.divider()
    st.caption("Payments are securely processed via SadaBiz. Your data is 100% private.")

# Simple logic to toggle the view
if not st.session_state.get('paid'):
    show_payment_page()
else:
    st.write("Welcome to the Pro dashboard!")