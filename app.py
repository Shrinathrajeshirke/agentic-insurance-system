import streamlit as st
import requests
import uuid

st.set_page_config(page_title="Term Life Policy Advisor", page_icon="🛡️", layout="wide")
st.title("🛡️ Term Life Insurance Advisory Engine")

# Persistent session tracking
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_profile" not in st.session_state:
    st.session_state.user_profile = None
if "onboarded" not in st.session_state:
    st.session_state.onboarded = False

# Sidebar for Ingestion and Session Controls
with st.sidebar:
    st.header("Upload Policy Brochure")
    st.caption("Permanently index official insurance contract PDFs into Qdrant.")
    
    with st.form("brochure_upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Choose Policy PDF", type=["pdf"])
        doc_policy_name = st.text_input("Policy Name", placeholder="e.g., Smart Term Plan")
        doc_insurer = st.text_input("Insurer Name", placeholder="e.g., Tata AIA")
        submit_doc = st.form_submit_button("Index Document into Qdrant")

        if submit_doc:
            if not uploaded_file or not doc_policy_name or not doc_insurer:
                st.error("Please fill in all document fields.")
            else:
                with st.spinner("Chunking and indexing into Qdrant..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        data = {"policy_name": doc_policy_name, "insurer": doc_insurer}
                        res = requests.post("http://localhost:8000/upload_brochure", files=files, data=data)
                        res.raise_for_status()
                        st.success(res.json().get("message", "Document indexed!"))
                    except Exception as err:
                        st.error(f"Failed to ingest document: {err}")

    st.divider()
    st.header("Session Status")
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
    if st.session_state.onboarded:
        st.success("Profile: Active & Verified")
        if st.button("Start New Session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.onboarded = False
            st.session_state.user_profile = None
            st.session_state.messages = []
            st.rerun()
    else:
        st.warning("Profile: Incomplete")

# Step 1: Onboarding Assessment
if not st.session_state.onboarded:
    st.subheader("Step 1: Underwriting Assessment")
    st.markdown("Provide your profile details to receive your top 3 policy recommendations.")

    with st.form("onboarding_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=80, value=28, step=1)
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
            annual_income_range = st.selectbox(
                "Annual Income Range",
                options=[
                    "Below 3 Lakhs",
                    "3 - 6 Lakhs",
                    "6 - 10 Lakhs",
                    "10 - 15 Lakhs",
                    "15 - 25 Lakhs",
                    "25+ Lakhs"
                ],
                index=2
            )
            is_smoker = st.radio("Tobacco/Nicotine User?", options=["No", "Yes"])

        with col2:
            dependents = st.number_input("Financial Dependents", min_value=0, max_value=10, value=1, step=1)
            sum_assured_range = st.selectbox(
                "Desired Life Cover (Sum Assured)",
                options=[
                    "50 Lakhs",
                    "75 Lakhs",
                    "1 Crore",
                    "1.5 Crore",
                    "2 Crore",
                    "2.5 Crore",
                    "3 Crore",
                    "5 Crore+"
                ],
                index=2
            )
            policy_term = st.number_input("Desired Policy Term (Years)", min_value=5, max_value=65, value=35, step=1)
            conditions = st.text_input("Medical Conditions", placeholder="e.g., None, Hypertension")

        submit_profile = st.form_submit_button("Generate Top 3 Recommendations")

        if submit_profile:
            med_list = [c.strip() for c in conditions.split(",") if c.strip()] if conditions else ["None"]
            profile_data = {
                "age": int(age),
                "gender": gender,
                "annual_income": annual_income_range,
                "is_smoker": True if is_smoker == "Yes" else False,
                "medical_conditions": med_list,
                "dependents_count": int(dependents),
                "desired_sum_assured": sum_assured_range,
                "policy_term_years": int(policy_term)
            }
            st.session_state.user_profile = profile_data

            # Explicit synthetic trigger for Indian market term policies
            initial_query = (
                "Initial profile assessment: Recommend the top 3 specific Indian term insurance plans "
                f"(evaluating options such as HDFC Life Click 2 Protect Super, Max Life Smart Secure Plus, "
                f"Tata AIA Sampoorna Raksha Supreme, ICICI Prudential iProtect Smart) matching an income of "
                f"{annual_income_range} and desired cover of {sum_assured_range}."
            )
            st.session_state.messages.append({"role": "user", "content": initial_query})

            with st.spinner("Finding best-fit Indian policy contracts..."):
                try:
                    payload = {
                        "session_id": st.session_state.session_id,
                        "messages": st.session_state.messages,
                        "user_profile": st.session_state.user_profile
                    }
                    res = requests.post("http://localhost:8000/chat", json=payload)
                    res.raise_for_status()
                    data = res.json()

                    st.session_state.messages.append({"role": "assistant", "content": data["reply"]})
                    st.session_state.user_profile = data["user_profile"]
                    st.session_state.onboarded = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error launching advisor: {e}")

# Step 2: Chat Interface
else:
    for msg in st.session_state.messages:
        if "Initial profile assessment:" in msg["content"]:
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about riders, exclusions, or compare benefits..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Analyzing policy clauses..."):
            try:
                payload = {
                    "session_id": st.session_state.session_id,
                    "messages": st.session_state.messages,
                    "user_profile": st.session_state.user_profile
                }
                res = requests.post("http://localhost:8000/chat", json=payload)
                res.raise_for_status()
                data = res.json()

                st.session_state.messages.append({"role": "assistant", "content": data["reply"]})
                st.session_state.user_profile = data["user_profile"]
                with st.chat_message("assistant"):
                    st.markdown(data["reply"])

            except Exception as e:
                st.error(f"Communication Error: {e}")