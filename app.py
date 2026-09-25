import streamlit as st
import requests
import json
import uuid
import pandas as pd
import os

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Term Life Policy Advisor", page_icon="🛡️", layout="wide")

# ==========================================
# 0. Session State Initialization
# ==========================================
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_profile" not in st.session_state:
    st.session_state.user_profile = None
if "onboarded" not in st.session_state:
    st.session_state.onboarded = False

def get_auth_headers():
    return {"Authorization": f"Bearer {st.session_state.auth_token}"}

def stream_chat_response(payload: dict):
    """
    Generator yielding (token, citations) from FastAPI SSE stream.
    Tokens stream in real-time; citations are emitted once retrieved.
    """
    citations = []
    try:
        with requests.post(
            f"{API_BASE}/chat/stream",
            json=payload,
            headers=get_auth_headers(),
            stream=True,
            timeout=60
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        try:
                            data = json.loads(data_str)
                            if "citations" in data:
                                citations = data["citations"]
                            elif "token" in data:
                                yield data["token"], None
                            elif "error" in data:
                                yield f"\n\n*Error: {data['error']}*", None
                                break
                            elif data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        yield f"\n\n*Connection error during generation: {e}*", None

    yield "", citations

# Grounded IRDAI benchmark comparison table data
COMPARISON_TABLE_DATA = [
    {
        "Plan Name": "HDFC Life Click 2 Protect Super",
        "Insurer": "HDFC Life",
        "Claim Settlement (Count)": "99.30%",
        "Amount Settlement (ASR)": "95.10%",
        "Critical Illnesses": "Up to 60 conditions",
        "Zero-Cost Exit (SEV)": "Available at specified age",
        "Grace Period": "30 days (Ann/Half/Qtr), 15 days (Mo)",
        "Suicide Exclusion": "12 months"
    },
    {
        "Plan Name": "Max Life Smart Secure Plus",
        "Insurer": "Max Life",
        "Claim Settlement (Count)": "99.65%",
        "Amount Settlement (ASR)": "96.40%",
        "Critical Illnesses": "Up to 64 conditions",
        "Zero-Cost Exit (SEV)": "Special Exit Value at age 65",
        "Grace Period": "30 days (Ann/Half/Qtr), 15 days (Mo)",
        "Suicide Exclusion": "12 months"
    },
    {
        "Plan Name": "Tata AIA Sampoorna Raksha Supreme",
        "Insurer": "Tata AIA",
        "Claim Settlement (Count)": "99.13%",
        "Amount Settlement (ASR)": "94.80%",
        "Critical Illnesses": "Up to 40 conditions",
        "Zero-Cost Exit (SEV)": "Available on select variants",
        "Grace Period": "30 days (Ann/Half/Qtr), 15 days (Mo)",
        "Suicide Exclusion": "12 months"
    },
    {
        "Plan Name": "ICICI Pru iProtect Smart",
        "Insurer": "ICICI Prudential",
        "Claim Settlement (Count)": "98.90%",
        "Amount Settlement (ASR)": "94.20%",
        "Critical Illnesses": "Up to 34 conditions",
        "Zero-Cost Exit (SEV)": "Smart Exit Benefit available",
        "Grace Period": "30 days (Ann/Half/Qtr), 15 days (Mo)",
        "Suicide Exclusion": "12 months"
    }
]

# ==========================================
# 1. AUTHENTICATION VIEW (Logged Out)
# ==========================================
if not st.session_state.auth_token:
    st.title("🛡️ AI Term Life Insurance Advisor")
    st.markdown("Secure, IRDAI-compliant policy recommendations with verifiable underwriting citations.")

    tab_login, tab_signup = st.tabs(["Log In", "Create Account"])

    with tab_login:
        with st.form("login_form"):
            login_email = st.text_input("Email")
            login_pass = st.text_input("Password", type="password")
            btn_login = st.form_submit_button("Sign In", width="stretch")

            if btn_login:
                try:
                    res = requests.post(f"{API_BASE}/auth/login", json={"email": login_email, "password": login_pass})
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.auth_token = data["access_token"]
                        st.session_state.user_email = data["email"]
                        st.rerun()
                    else:
                        st.error(res.json().get("detail", "Authentication failed."))
                except Exception as e:
                    st.error(f"Connection error: {e}")

    with tab_signup:
        with st.form("signup_form"):
            signup_email = st.text_input("Email")
            signup_pass = st.text_input("Password", type="password")
            btn_signup = st.form_submit_button("Create Account", width="stretch")

            if btn_signup:
                try:
                    res = requests.post(f"{API_BASE}/auth/register", json={"email": signup_email, "password": signup_pass})
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.auth_token = data["access_token"]
                        st.session_state.user_email = data["email"]
                        st.success("Account created successfully!")
                        st.rerun()
                    else:
                        st.error(res.json().get("detail", "Registration failed."))
                except Exception as e:
                    st.error(f"Connection error: {e}")

# ==========================================
# 2. ADVISORY VIEW (Logged In)
# ==========================================
else:
    with st.sidebar:
        st.write(f"Logged in as: **{st.session_state.user_email}**")
        if st.button("Log Out", width="stretch"):
            st.session_state.auth_token = None
            st.session_state.user_email = None
            st.session_state.current_thread_id = None
            st.session_state.messages = []
            st.session_state.user_profile = None
            st.session_state.onboarded = False
            st.rerun()

        st.divider()

        if st.button("➕ New Chat", width="stretch", type="primary"):
            try:
                res = requests.post(f"{API_BASE}/threads/new", headers=get_auth_headers())
                if res.status_code == 200:
                    thread_data = res.json()
                    st.session_state.current_thread_id = thread_data["id"]
                    st.session_state.messages = []
                    st.session_state.user_profile = None
                    st.session_state.onboarded = False
                    st.rerun()
            except Exception as e:
                st.error(f"Error creating session: {e}")

        st.subheader("Your Chat Sessions")
        try:
            res = requests.get(f"{API_BASE}/threads", headers=get_auth_headers())
            if res.status_code == 200:
                threads = res.json()
                for t in threads:
                    t_id = t["id"]
                    t_title = t["title"]
                    is_active = (t_id == st.session_state.current_thread_id)

                    col_btn, col_opt = st.columns([5, 1])
                    label = f"{'🟢 ' if is_active else ''}{t_title}"

                    if col_btn.button(label, key=f"select_{t_id}", width="stretch"):
                        st.session_state.current_thread_id = t_id
                        st.session_state.messages = []
                        st.session_state.onboarded = True
                        st.rerun()

                    with col_opt.popover("⚙️"):
                        st.markdown(f"**Manage:** *{t_title}*")
                        new_name = st.text_input("New title", value=t_title, key=f"rename_{t_id}")
                        if st.button("Save Name", key=f"save_{t_id}", width="stretch"):
                            if new_name.strip():
                                r = requests.patch(
                                    f"{API_BASE}/threads/{t_id}/rename",
                                    json={"title": new_name.strip()},
                                    headers=get_auth_headers()
                                )
                                if r.status_code == 200:
                                    st.success("Renamed!")
                                    st.rerun()
                                else:
                                    st.error("Failed to rename.")

                        st.divider()
                        if st.button("🗑️ Delete Chat", key=f"del_{t_id}", type="secondary", width="stretch"):
                            r = requests.delete(f"{API_BASE}/threads/{t_id}", headers=get_auth_headers())
                            if r.status_code == 200:
                                if st.session_state.current_thread_id == t_id:
                                    st.session_state.current_thread_id = None
                                    st.session_state.messages = []
                                    st.session_state.user_profile = None
                                    st.session_state.onboarded = False
                                st.rerun()
                            else:
                                st.error("Failed to delete.")
        except Exception as e:
            st.caption(f"Unable to fetch sessions: {e}")

        st.divider()

        st.subheader("Upload Policy Brochure")
        with st.form("brochure_upload_form", clear_on_submit=True):
            uploaded_file = st.file_uploader("Policy PDF", type=["pdf"])
            doc_policy_name = st.text_input("Policy Name", placeholder="e.g., Click 2 Protect Super")
            doc_insurer = st.text_input("Insurer Name", placeholder="e.g., HDFC Life")
            submit_doc = st.form_submit_button("Index into Qdrant", width="stretch")

            if submit_doc:
                if not uploaded_file or not doc_policy_name or not doc_insurer:
                    st.error("Please fill in all document fields.")
                else:
                    with st.spinner("Chunking & Indexing with Page Tracking..."):
                        try:
                            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                            data = {"policy_name": doc_policy_name, "insurer": doc_insurer}
                            res = requests.post(f"{API_BASE}/upload_brochure", files=files, data=data)
                            res.raise_for_status()
                            st.success(res.json().get("message", "Document indexed successfully!"))
                        except Exception as err:
                            st.error(f"Ingestion failed: {err}")

    if not st.session_state.current_thread_id:
        try:
            res = requests.post(f"{API_BASE}/threads/new", headers=get_auth_headers())
            if res.status_code == 200:
                st.session_state.current_thread_id = res.json()["id"]
        except Exception:
            st.session_state.current_thread_id = str(uuid.uuid4())

    st.title("🛡️ Term Life Insurance Advisory Engine")

    # ==========================================
    # Step A: Onboarding Wizard (New Session)
    # ==========================================
    if not st.session_state.onboarded:
        st.subheader("Underwriting Assessment")
        st.markdown("Enter your eligibility profile to initiate policy contract retrieval and benchmarking.")

        with st.form("onboarding_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=18, max_value=80, value=28, step=1)
                gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
                annual_income_range = st.selectbox(
                    "Annual Income Range",
                    options=["Below 3 Lakhs", "3 - 6 Lakhs", "6 - 10 Lakhs", "10 - 15 Lakhs", "15 - 25 Lakhs", "25+ Lakhs"],
                    index=2
                )
                is_smoker = st.radio("Tobacco / Nicotine User?", options=["No", "Yes"])

            with col2:
                dependents = st.number_input("Financial Dependents", min_value=0, max_value=10, value=1, step=1)
                sum_assured_range = st.selectbox(
                    "Desired Life Cover (Sum Assured)",
                    options=["50 Lakhs", "75 Lakhs", "1 Crore", "1.5 Crore", "2 Crore", "2.5 Crore", "3 Crore", "5 Crore+"],
                    index=2
                )
                policy_term = st.number_input("Policy Term (Years)", min_value=5, max_value=65, value=35, step=1)
                conditions = st.text_input("Medical Conditions", placeholder="e.g., None, Hypertension, Diabetes")

            submit_profile = st.form_submit_button("Generate Top 3 Recommendations", width="stretch")

        if submit_profile:
            med_list = [c.strip() for c in conditions.split(",") if c.strip()] if conditions else ["None"]
            st.session_state.user_profile = {
                "age": int(age),
                "gender": gender,
                "annual_income": annual_income_range,
                "is_smoker": True if is_smoker == "Yes" else False,
                "medical_conditions": med_list,
                "dependents_count": int(dependents),
                "desired_sum_assured": sum_assured_range,
                "policy_term_years": int(policy_term)
            }

            initial_query = (
                "Initial profile assessment: Recommend the top 3 specific Indian term insurance plans "
                f"(evaluating options such as HDFC Life Click 2 Protect Super, Max Life Smart Secure Plus, "
                f"Tata AIA Sampoorna Raksha Supreme, ICICI Prudential iProtect Smart) matching an income of "
                f"{annual_income_range} and desired cover of {sum_assured_range}."
            )
            st.session_state.messages.append({"role": "user", "content": initial_query})

            payload = {
                "thread_id": st.session_state.current_thread_id,
                "messages": st.session_state.messages,
                "user_profile": st.session_state.user_profile
            }

            with st.chat_message("assistant"):
                token_placeholder = st.empty()
                collected_tokens = []
                received_citations = []

                for token, citations in stream_chat_response(payload):
                    if token:
                        collected_tokens.append(token)
                        token_placeholder.markdown("".join(collected_tokens))
                    if citations:
                        received_citations = citations

                full_reply = "".join(collected_tokens)
                if received_citations:
                    with st.expander("🔍 Verified Contract Evidence & Citations", expanded=False):
                        for c in received_citations:
                            insurer_name = c.get("insurer") or "Policy Document"
                            policy_name = c.get("policy_name") or c.get("insurer") or "Term Insurance Plan"
                            page_val = c.get("page", "N/A")
                            section_val = c.get("section", "Contract Clause")
                            snippet_val = c.get("snippet", "")
                            st.markdown(
                                f"**{insurer_name} — {policy_name}** (Page {page_val} | *{section_val}*)\n"
                                f"> *\"{snippet_val}\"*\n"
                            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_reply,
                "citations": received_citations
            })
            st.session_state.onboarded = True
            st.rerun()

    # ==========================================
    # Step B: Interactive Chat Transcript + Matrix + PDF Export
    # ==========================================
    else:
        # Comparison Matrix Expander
        with st.expander("📊 Side-by-Side Policy Comparison Matrix (IRDAI Benchmarks)", expanded=False):
            df = pd.DataFrame(COMPARISON_TABLE_DATA)
            st.dataframe(df, width="stretch", hide_index=True)
            st.caption("Data derived from IRDAI Annual Disclosures and respective Public Filing Brochure documents.")

        # Downloadable PDF Report Button
        first_assistant_msg = next((m["content"] for m in st.session_state.messages if m["role"] == "assistant"), None)
        if first_assistant_msg and st.session_state.user_profile:
            col_pdf1, col_pdf2 = st.columns([4, 1])
            with col_pdf2:
                try:
                    pdf_payload = {
                        "user_profile": st.session_state.user_profile,
                        "advisory_text": first_assistant_msg
                    }
                    pdf_res = requests.post(
                        f"{API_BASE}/export_advisory_pdf",
                        json=pdf_payload,
                        headers=get_auth_headers(),
                        timeout=30
                    )
                    if pdf_res.status_code == 200:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_res.content,
                            file_name="term_life_advisory_brief.pdf",
                            mime="application/pdf",
                            width="stretch"
                        )
                except Exception:
                    pass

        # Chat Transcript
        for msg in st.session_state.messages:
            if "initial profile assessment" in msg["content"].lower():
                continue

            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                if msg.get("citations"):
                    with st.expander("🔍 Verified Contract Evidence & Citations", expanded=False):
                        for c in msg["citations"]:
                            insurer_name = c.get("insurer") or "Policy Document"
                            policy_name = c.get("policy_name") or c.get("insurer") or "Term Insurance Plan"
                            page_val = c.get("page", "N/A")
                            section_val = c.get("section", "Contract Clause")
                            snippet_val = c.get("snippet", "")
                            st.markdown(
                                f"**{insurer_name} — {policy_name}** (Page {page_val} | *{section_val}*)\n"
                                f"> *\"{snippet_val}\"*\n"
                            )

        if prompt := st.chat_input("Ask about riders, exclusions, claim settlement, or return of premium..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            payload = {
                "thread_id": st.session_state.current_thread_id,
                "messages": st.session_state.messages,
                "user_profile": st.session_state.user_profile
            }

            with st.chat_message("assistant"):
                token_placeholder = st.empty()
                collected_tokens = []
                received_citations = []

                for token, citations in stream_chat_response(payload):
                    if token:
                        collected_tokens.append(token)
                        token_placeholder.markdown("".join(collected_tokens))
                    if citations:
                        received_citations = citations

                full_reply = "".join(collected_tokens)
                if received_citations:
                    with st.expander("🔍 Verified Contract Evidence & Citations", expanded=False):
                        for c in received_citations:
                            insurer_name = c.get("insurer") or "Policy Document"
                            policy_name = c.get("policy_name") or c.get("insurer") or "Term Insurance Plan"
                            page_val = c.get("page", "N/A")
                            section_val = c.get("section", "Contract Clause")
                            snippet_val = c.get("snippet", "")
                            st.markdown(
                                f"**{insurer_name} — {policy_name}** (Page {page_val} | *{section_val}*)\n"
                                f"> *\"{snippet_val}\"*\n"
                            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_reply,
                "citations": received_citations
            })