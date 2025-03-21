import streamlit as st
import pandas as pd
import plotly.express as px
import datetime

# Define the data directory (use raw string to avoid escape issues)
data_dir = r"D:\Benson\aUpWork\Hussein Wael Egypt\Robo Advisory Platform\Portfolios"

#############################################
# 1. RISK CALCULATION FUNCTIONS & DATA
#############################################

# QUESTION WEIGHTS (Sum = 1.0)
question_weights = {
    "Q1": 0.125,  # Purpose of Investment
    "Q2": 0.175,  # Time Horizon
    "Q3": 0.0,    # Type of Investor (No impact on risk score)
    "Q4": 0.45,   # Reaction to Loss (Most Influential Factor)
    "Q5": 0.25,   # Monthly Investment Percentage
}

# ANSWER VALUES (0.0 = LOWEST RISK, 1.0 = HIGHEST)
answer_values = {
    "Q1": {"A": 0.0, "B": 0.33, "C": 0.67, "D": 1.0},
    "Q2": {"A": 0.0, "B": 0.33, "C": 0.67, "D": 1.0},
    "Q3": {"A": 0.0, "B": 0.0, "C": 0.0},  # No impact on risk score
    "Q4": {"A": 0.0, "B": 0.33, "C": 0.67, "D": 1.0},
    "Q5": {"A": 0.0, "B": 0.33, "C": 0.67, "D": 0.8, "E": 1.0},  # Adjusted to ensure separation
}

# AGE-BASED RISK LIMIT
def get_max_risk_by_age(age):
    if age >= 50:
        return 7.0
    elif 41 <= age <= 50:
        return 8.0
    elif 33 <= age <= 40:
        return 9.0
    elif 26 <= age <= 32:
        return 10.0
    else:
        return 9.5

# AGE-BASED RISK FACTOR
def get_age_risk_factor(age):
    if 18 <= age <= 25:
        return 1.1
    elif 26 <= age <= 32:
        return 1.2
    elif 33 <= age <= 40:
        return 1.0
    elif 41 <= age <= 50:
        return 0.9
    else:
        return 0.8

# MULTI-QUESTION INCONSISTENCY DETECTION
def detect_inconsistencies(user_answers, age):
    contradiction_score = 0
    detected_issues = []

    # 1) Emergency Fund Users Should Not Invest Aggressively
    if user_answers["Q1"] == "A" and user_answers["Q2"] in ["B", "C", "D"]:
        if user_answers["Q4"] == "D" and user_answers["Q5"] in ["D", "E"]:
            if user_answers["Q5"] == "E":
                contradiction_score += 2
                detected_issues.append("Emergency fund users should not invest more than 50% in aggressive markets.")
            else:
                contradiction_score += 1
                detected_issues.append("Emergency fund users should not invest 30%-50% in aggressive markets.")

    # 2) Short-Term Investors Should Not Take High Risks
    if user_answers["Q2"] == "A" and user_answers["Q4"] == "D" and user_answers["Q5"] in ["D", "E"]:
        if user_answers["Q5"] == "E":
            contradiction_score += 2
            detected_issues.append("Short-term investors should not allocate more than 50% aggressively.")
        else:
            contradiction_score += 1
            detected_issues.append("Short-term investors should not allocate 30%-50% aggressively.")

    # 3) Long-Term Investors Should Not Panic Sell
    if user_answers["Q2"] == "D" and user_answers["Q4"] == "A" and user_answers["Q5"] in ["D", "E"]:
        if user_answers["Q5"] == "E":
            contradiction_score += 2
            detected_issues.append("Long-term investors should not panic sell with high allocations (50%+).")
        else:
            contradiction_score += 1
            detected_issues.append("Long-term investors should not panic sell with moderate-high allocations (30%-50%).")

    # 4) Retirement Investing Should Not Be Short-Term
    if user_answers["Q1"] == "B" and user_answers["Q2"] == "A" and user_answers["Q5"] in ["D", "E"]:
        if user_answers["Q5"] == "E":
            contradiction_score += 2
            detected_issues.append("Retirement investing should not involve more than 50% aggressive investing.")
        else:
            contradiction_score += 1
            detected_issues.append("Retirement investing should not involve 30%-50% aggressive investing.")

    # 5) Older Investors (50+) Should Avoid Extreme Risk
    if age >= 50 and user_answers["Q5"] in ["D", "E"]:
        if user_answers["Q5"] == "E":
            contradiction_score += 2
            detected_issues.append("Investors above 50 should avoid investing more than 50% of their income aggressively.")
        else:
            contradiction_score += 1
            detected_issues.append("Older investors should reconsider investing 30-50% aggressively.")

    # 6) Young Investors (<30) Should Not Be Too Conservative
    if age <= 30 and user_answers["Q5"] == "A" and user_answers["Q2"] == "D":
        detected_issues.append("Younger investors with long-term goals should consider higher risk for potential growth.")

    # 7) Middle-Aged Investors (40-50) Should Not Panic Sell
    if 41 <= age <= 50 and user_answers["Q2"] == "D" and user_answers["Q4"] == "A":
        detected_issues.append("Long-term investors should not panic sell after short-term losses.")

    # 8) Retirement Planning Should Have a Long Horizon
    if user_answers["Q1"] == "B" and user_answers["Q2"] == "A":
        detected_issues.append("Retirement investing should have a longer horizon to allow for compounding growth.")

    # 9) High-Risk Investing Should Align with Time Horizon
    if user_answers["Q2"] == "A" and user_answers["Q5"] == "E":
        detected_issues.append("Short-term investments should avoid aggressive allocation to reduce volatility risk.")

    return contradiction_score, detected_issues

# CONFIDENCE SCORE & PROBABILITY SYSTEM
def calculate_confidence_score(inconsistency_score, user_answers):
    max_inconsistencies = 5.0
    inconsistency_rate = inconsistency_score / max_inconsistencies
    base_confidence = (1 - inconsistency_rate) * 100

    risk_values = [answer_values[q][user_answers[q]] for q in user_answers]
    risk_variability = max(risk_values) - min(risk_values)
    risk_factor = 1 - 0.2 * risk_variability
    adjusted_confidence = base_confidence * risk_factor

    return round(max(adjusted_confidence, 0), 1)

# INVESTOR PROFILE CLASSIFICATION
def assign_investor_profile(score):
    if score <= 2.5:
        return "Guardian 🛡"
    elif score <= 4.0:
        return "Strategic Planner 🏗"
    elif score <= 6.5:
        return "Growth Seeker 📈"
    elif score <= 8.0:
        return "Dynamic Investor 🚀"
    elif score <= 9.5:
        return "Bold Visionary 🌍"
    else:
        return "Maximizer 🎯"

# FINAL RISK SCORE COMPUTATION (with All "A" Rule)
def calculate_final_risk_score(user_answers, age):
    # Special rule: If all answers are "A", override with a custom score by age
    if all(answer == "A" for answer in user_answers.values()):
        if 18 <= age <= 30:
            final_score = 3.5
        elif 31 <= age <= 40:
            final_score = 3.0
        elif 41 <= age <= 49:
            final_score = 2.5
        else:
            final_score = 2.0
        confidence = calculate_confidence_score(0, user_answers)
        return final_score, confidence, []

    age_factor = get_age_risk_factor(age)
    max_risk = get_max_risk_by_age(age)

    # Base risk score from weighted answers
    base_score = sum([
        question_weights[q] * answer_values[q][user_answers[q]]
        for q in user_answers
    ])

    # Check for inconsistencies
    inconsistency_score, issues = detect_inconsistencies(user_answers, age)

    # Map base_score to an age-adjusted scale
    mapped_score = max(
        2.0 * age_factor,
        2.0 + (age_factor - 1.0) + base_score * (max_risk - 2.0)
    )

    # Subtract contradiction score
    mapped_score -= inconsistency_score

    # Enforce boundaries
    adjusted_score = max(2.0 * age_factor, min(mapped_score, max_risk))
    final_rounded_score = round(adjusted_score * 2) / 2

    confidence = calculate_confidence_score(inconsistency_score, user_answers)
    return final_rounded_score, confidence, issues


#############################################
# 2. DATA IMPORT FUNCTIONS (Using data_dir)
#############################################

@st.cache_data
def load_portfolio_weights():
    # Read the Excel file for portfolio weights using data_dir
    df = pd.read_excel(f"{data_dir}/Sector Portfolio Weights Geom_Mapping.xlsx")
    
    # Remove leading/trailing spaces from column names
    df.columns = df.columns.str.strip()
    
    # Ensure 'Risk Score' column actually exists
    if 'Risk Score' not in df.columns:
        st.error("Your Excel file must have a column named 'Risk Score'. Please fix the file.")
        return pd.DataFrame()  # Return empty, so the code won't break further
    
    return df

@st.cache_data
def get_portfolio_weights(df_weights, risk_score):
    # If df_weights is empty or missing 'Risk Score', return {}
    if df_weights.empty or 'Risk Score' not in df_weights.columns:
        return {}
    
    # Return a dictionary of sector weights for the given risk_score (exact match)
    row = df_weights.loc[df_weights['Risk Score'] == risk_score]
    if row.empty:
        # If no exact match, return {}
        return {}
    row_dict = row.drop(columns=['Risk Score']).to_dict(orient='records')[0]
    return row_dict

@st.cache_data
def load_sector_returns():
    # Read the Excel file for sector daily returns using data_dir
    df = pd.read_excel(f"{data_dir}/Sector_Daily_Returns.xlsx", parse_dates=['Date'])
    
    # Remove leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # ---- NEW LINE: Drop all rows containing NA values ----
    df.dropna(inplace=True)
    
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    return df

def compute_portfolio_cumulative_return(df_returns, weights_dict):
    # If no weights, return empty
    if not weights_dict:
        return pd.Series([], dtype=float)
    
    weights_s = pd.Series(weights_dict)
    # Intersection of columns in df_returns with keys in weights_dict
    common_sectors = list(set(df_returns.columns).intersection(weights_s.index))
    if not common_sectors:
        return pd.Series([], dtype=float)
    
    df_sub = df_returns[common_sectors]
    # Reorder weights_s to match df_sub columns
    weights_s = weights_s[df_sub.columns]
    
    # daily_portfolio_return = sum of (sector_return * sector_weight)
    daily_portfolio_return = df_sub.dot(weights_s)
    # cumulative returns = (1 + daily_return).cumprod() - 1
    cum_returns = (1 + daily_portfolio_return).cumprod() - 1
    return cum_returns

def filter_cumulative_return_by_period(cum_returns, period):
    if len(cum_returns) == 0:
        return cum_returns
    
    end_date = cum_returns.index[-1]
    if period == 'ALL':
        start_date = cum_returns.index[0]
    elif period == '1Y':
        start_date = end_date - pd.DateOffset(years=1)
    elif period == '3Y':
        start_date = end_date - pd.DateOffset(years=3)
    elif period == '5Y':
        start_date = end_date - pd.DateOffset(years=5)
    elif period == '10Y':
        start_date = end_date - pd.DateOffset(years=10)
    else:
        start_date = cum_returns.index[0]
    filtered = cum_returns.loc[cum_returns.index >= start_date]
    return filtered


#############################################
# 3. APP STATE INITIALIZATION
#############################################

if "page" not in st.session_state:
    st.session_state.page = 1  # Pages 1-7
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "age" not in st.session_state:
    st.session_state.age = None
if "final_score" not in st.session_state:
    st.session_state.final_score = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "issues" not in st.session_state:
    st.session_state.issues = []
if "investor_profile" not in st.session_state:
    st.session_state.investor_profile = ""

#############################################
# 4. ORIGINAL QUESTION TEXT (VERBATIM)
#############################################

# Q1
q1_header_html = "<h3>1) What Are You Saving or Investing For?</h3>"
q1_tip_html = (
    "<p><b>Tip:</b> Think about your main goal for this portfolio—"
    "imagine where you want to be in the near future. Is this portfolio your safety net, "
    "your dream home fund, or part of your long-term wealth-building plan?</p>"
)
q1_options = [
    ("An emergency safety net 🏦", "A"),
    ("Saving for retirement 👵", "B"),
    ("A large purchase (home, education, etc.) 🏠", "C"),
    ("Building wealth over time 🌱", "D"),
]
q1_why_html = (
    "<p><b>Why Is It Important?</b><br>"
    "Your goal helps us figure out how much risk you can handle and how to set your priorities. "
    "For example, growing money for a house in 10 years might mean taking a bit more risk, "
    "while keeping cash for emergencies needs to stay safe and easy to reach.</p>"
)

# Q2
q2_header_html = "<h3>2) When Do You Think You'll Need This Money?</h3>"
q2_tip_html = (
    "<p><b>Tip:</b> Think of this portfolio like a roadmap. Is this a quick stop or a long journey? "
    "Maybe you'll need it soon—or maybe it's far off. If it's far off, we can take more chances to grow it bigger over time.</p>"
)
q2_options = [
    ("Less than 3 years ⏱", "A"),
    ("3–5 years ⌛", "B"),
    ("6–10 years 📅", "C"),
    ("More than 10 years 🗓", "D"),
]
q2_why_html = (
    "<p><b>Why Is It Important?</b><br>"
    "How soon you need your money changes how risky your investments can be. "
    "If it's soon, we'll keep it safer. If it's far off, we can take more chances to grow your investment.</p>"
)

# Q3
q3_header_html = "<h3>3) What Kind of Investor Do You Want to Be?</h3>"
q3_tip_html = (
    "<p><b>Tip:</b> Every portfolio can reflect a personal style. Some people prefer steady growth; "
    "others want something more dynamic or value-driven. Your choice here will guide how your money grows "
    "and what it stands for.</p>"
)
q3_options = [
    ("Steady Growth (Balanced, long-term focus) 🌊", "A"),
    ("Value-Driven (Sustainable, ethical, Shariah-compliant) 🍃", "B"),
    ("Smart & Strategic (mix of innovation & stability) ⚙️", "C"),
]
q3_why_html = (
    "<p><b>Why Is It Important?</b><br>"
    "This helps us design a portfolio that matches your vision. It's about building a future you believe in, "
    "with investments that feel right every step of the way. Remember, you can build multiple portfolios "
    "with different strategies.</p>"
)

# Q4
q4_header_html = (
    "<h3>4) Imagine You Started With EGP 100,000, Then Lost EGP 10,000 After One Month. What Next?</h3>"
)
q4_tip_html = (
    "<p><b>Tip:</b> Your instinct shows how you handle market swings. There's no right or wrong—just be honest "
    "about how you feel.</p>"
)
q4_options = [
    ("I would sell all 😰", "A"),
    ("I would sell some 🤔", "B"),
    ("I would hold 🧐", "C"),
    ("I would buy more 🤑", "D"),
]
q4_why_html = (
    "<p><b>Why Is It Important?</b><br>"
    "How you react to risk helps us shape a strategy that fits you. No judgment—just a smarter plan "
    "based on your comfort level.</p>"
)

# Q5
q5_header_html = "<h3>5) How Much of Your Monthly Income Are You Comfortable Investing?</h3>"
q5_tip_html = (
    "<p><b>Tip:</b> Think about your monthly income after paying for necessities like rent and groceries. "
    "How much of what's left can you comfortably invest without causing financial stress?</p>"
)
q5_options = [
    ("Less than 10% (I'm just getting started) 🌱", "A"),
    ("10–20% (A balanced approach) 🏗", "B"),
    ("20–30% (I'm ready to commit more) 🚀", "C"),
    ("30–50% (I'm comfortable investing a large share) 🌕", "D"),
    ("More than 50% (I'm all in—invest aggressively) 🔥", "E"),
]
q5_why_html = (
    "<p><b>Why Is It Important?</b><br>"
    "Consistent investing leads to long-term success. Knowing how much you can invest each month "
    "helps us plan a strategy that fits your budget. Investing within your means promotes stability "
    "and builds your investment over time.</p>"
)

# Final Check
final_check_header_html = "<h3>Final Check: Does This Sound Right?</h3>"
final_check_text_html = (
    "<p>“Before we build your portfolio, let’s make sure everything looks good!”</p>"
)

#############################################
# 5. PAGE RENDERING FUNCTIONS
#############################################

def page1():
    st.markdown("## Page 1: Investment Goal & Age")
    st.markdown("### Step 1 of 7")
    age = st.slider("Your Age:", min_value=18, max_value=90, value=35, step=1)
    st.session_state.age = age
    
    st.markdown(q1_header_html, unsafe_allow_html=True)
    st.markdown(q1_tip_html, unsafe_allow_html=True)
    
    current_q1 = st.session_state.user_answers.get("Q1", "A")
    labels = [item[0] for item in q1_options]
    values = [item[1] for item in q1_options]
    default_index = values.index(current_q1) if current_q1 in values else 0
    
    answer = st.radio(
        "Choose one:",
        options=values,
        format_func=lambda x: labels[values.index(x)],
        index=default_index
    )
    
    st.markdown(q1_why_html, unsafe_allow_html=True)
    
    if st.button("Next"):
        st.session_state.user_answers["Q1"] = answer
        st.session_state.page = 2
        st.rerun()

def page2():
    st.markdown("## Page 2: Time Horizon")
    st.markdown("### Step 2 of 7")
    st.markdown(q2_header_html, unsafe_allow_html=True)
    st.markdown(q2_tip_html, unsafe_allow_html=True)
    
    current_q2 = st.session_state.user_answers.get("Q2", "A")
    labels = [item[0] for item in q2_options]
    values = [item[1] for item in q2_options]
    default_index = values.index(current_q2) if current_q2 in values else 0
    
    answer = st.radio(
        "Choose one:",
        options=values,
        format_func=lambda x: labels[values.index(x)],
        index=default_index
    )
    
    st.markdown(q2_why_html, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            st.session_state.page = 1
            st.rerun()
    with col2:
        if st.button("Next"):
            st.session_state.user_answers["Q2"] = answer
            st.session_state.page = 3
            st.rerun()

def page3():
    st.markdown("## Page 3: Type of Investor")
    st.markdown("### Step 3 of 7")
    st.markdown(q3_header_html, unsafe_allow_html=True)
    st.markdown(q3_tip_html, unsafe_allow_html=True)
    
    current_q3 = st.session_state.user_answers.get("Q3", "A")
    labels = [item[0] for item in q3_options]
    values = [item[1] for item in q3_options]
    default_index = values.index(current_q3) if current_q3 in values else 0
    
    answer = st.radio(
        "Choose one:",
        options=values,
        format_func=lambda x: labels[values.index(x)],
        index=default_index
    )
    
    st.markdown(q3_why_html, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            st.session_state.page = 2
            st.rerun()
    with col2:
        if st.button("Next"):
            st.session_state.user_answers["Q3"] = answer
            st.session_state.page = 4
            st.rerun()

def page4():
    st.markdown("## Page 4: Reaction to Loss")
    st.markdown("### Step 4 of 7")
    st.markdown(q4_header_html, unsafe_allow_html=True)
    st.markdown(q4_tip_html, unsafe_allow_html=True)
    
    current_q4 = st.session_state.user_answers.get("Q4", "A")
    labels = [item[0] for item in q4_options]
    values = [item[1] for item in q4_options]
    default_index = values.index(current_q4) if current_q4 in values else 0
    
    answer = st.radio(
        "Choose one:",
        options=values,
        format_func=lambda x: labels[values.index(x)],
        index=default_index
    )
    
    st.markdown(q4_why_html, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            st.session_state.page = 3
            st.rerun()
    with col2:
        if st.button("Next"):
            st.session_state.user_answers["Q4"] = answer
            st.session_state.page = 5
            st.rerun()

def page5():
    st.markdown("## Page 5: Monthly Investment")
    st.markdown("### Step 5 of 7")
    st.markdown(q5_header_html, unsafe_allow_html=True)
    st.markdown(q5_tip_html, unsafe_allow_html=True)
    
    current_q5 = st.session_state.user_answers.get("Q5", "A")
    labels = [item[0] for item in q5_options]
    values = [item[1] for item in q5_options]
    default_index = values.index(current_q5) if current_q5 in values else 0
    
    answer = st.radio(
        "Choose one:",
        options=values,
        format_func=lambda x: labels[values.index(x)],
        index=default_index
    )
    
    st.markdown(q5_why_html, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            st.session_state.page = 4
            st.rerun()
    with col2:
        if st.button("Next"):
            st.session_state.user_answers["Q5"] = answer
            st.session_state.page = 6
            st.rerun()

def page6():
    st.markdown("## Page 6: Final Check")
    st.markdown("### Step 6 of 7")
    st.markdown(final_check_header_html, unsafe_allow_html=True)
    st.markdown(final_check_text_html, unsafe_allow_html=True)
    
    cb_goal = st.checkbox("My investment goal and time horizon are correct.", value=st.session_state.get("cb_goal", False))
    cb_risk = st.checkbox("My risk comfort level matches my choice.", value=st.session_state.get("cb_risk", False))
    cb_understand = st.checkbox("I understand that this will shape my portfolio strategy.", value=st.session_state.get("cb_understand", False))
    cb_done = st.checkbox("Done!", value=st.session_state.get("cb_done", False))
    
    st.session_state.cb_goal = cb_goal
    st.session_state.cb_risk = cb_risk
    st.session_state.cb_understand = cb_understand
    st.session_state.cb_done = cb_done
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            st.session_state.page = 5
            st.rerun()
    with col2:
        if st.button("Calculate Risk"):
            if not (cb_goal and cb_risk and cb_understand and cb_done):
                st.error("Please tick all checkboxes (the three statements plus 'Done!') before proceeding.")
            else:
                final_score, confidence, issues = calculate_final_risk_score(
                    st.session_state.user_answers, 
                    st.session_state.age
                )
                investor_profile = assign_investor_profile(final_score)
                
                st.session_state.final_score = final_score
                st.session_state.confidence = confidence
                st.session_state.issues = issues
                st.session_state.investor_profile = investor_profile
                
                st.session_state.page = 7
                st.rerun()

def page7():
    st.markdown("## Page 7: Portfolio & Results")
    st.markdown("### Final Risk Assessment")
    
    final_score = st.session_state.get("final_score", None)
    confidence = st.session_state.get("confidence", None)
    issues = st.session_state.get("issues", [])
    investor_profile = st.session_state.get("investor_profile", "")
    
    if final_score is None:
        st.warning("No risk assessment found. Please complete the questionnaire first.")
        return
    
    st.success("Risk Assessment Complete!")
    st.markdown(f"**Your Risk Score:** {final_score:.2f}")
    st.markdown(f"**Investor Type:** {investor_profile}")
    st.markdown(f"**Confidence Level:** {confidence:.1f}%")
    
    if issues:
        st.markdown("**Inconsistencies / Warnings:**")
        for issue in issues:
            st.warning(f"⚠ {issue}")
    else:
        st.markdown("✅ No inconsistencies detected.")
    
    # Slider to explore different risk scores
    st.markdown("---")
    st.markdown("### Customize Your Risk Score")
    st.markdown(
        "Use the slider below to explore different risk scores (0.5 to 10). "
        "Your assigned score is the default, but you can adjust it to see how the portfolio allocations change."
    )
    selected_risk_score = st.slider("Risk Score", min_value=0.5, max_value=10.0, value=float(final_score), step=0.5)
    
    # Load data & get weights
    df_weights = load_portfolio_weights()
    weights_dict = get_portfolio_weights(df_weights, selected_risk_score)
    
    st.markdown("### Portfolio Allocation")
    if not weights_dict:
        st.warning("No allocation data found for this risk score. Make sure 'Risk Score' exists in your Excel file.")
    else:
        allocation = []
        for sector, w in weights_dict.items():
            if w != 0:
                percent_val = round(w, 4) * 100
                allocation.append((sector, percent_val))
        
        if not allocation:
            st.warning("All sectors are 0% for this score—nothing to display.")
        else:
            allocation.sort(key=lambda x: x[1], reverse=True)
            df_alloc = pd.DataFrame(allocation, columns=["Sector", "Allocation (%)"])
            
            fig_alloc = px.bar(
                df_alloc,
                x="Allocation (%)",
                y="Sector",
                orientation="h",
                text="Allocation (%)",
                title="Sector Allocation"
            )
            fig_alloc.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig_alloc.update_layout(
                xaxis_title="Allocation (%)",
                yaxis_title="Sector",
                margin=dict(l=80, r=40, t=80, b=40)
            )
            st.plotly_chart(fig_alloc, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Historical Performance")
    st.markdown("Below is the simulated cumulative return of your portfolio based on the selected risk score.")
    
    period_choice = st.selectbox("Select Timeframe:", ["1Y", "3Y", "5Y", "10Y", "ALL"], index=1)
    df_returns = load_sector_returns()
    cum_returns = compute_portfolio_cumulative_return(df_returns, weights_dict)
    filtered_cum = filter_cumulative_return_by_period(cum_returns, period_choice)
    
    if len(filtered_cum) == 0:
        st.warning("Not enough data to display a chart for this risk score or timeframe.")
    else:
        final_cum_val = filtered_cum.iloc[-1] * 100
        st.markdown(f"**Cumulative Return:** {final_cum_val:.2f}% All time.")
        
        df_plot = filtered_cum.reset_index()
        df_plot.columns = ["Date", "Cumulative Return"]
        df_plot["Cumulative Return %"] = df_plot["Cumulative Return"] * 100
        
        fig_perf = px.line(
            df_plot,
            x="Date",
            y="Cumulative Return %",
            title="Cumulative Portfolio Returns",
            labels={"Cumulative Return %": "Cumulative Return (%)"}
        )
        fig_perf.update_layout(
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode="x unified"
        )
        st.plotly_chart(fig_perf, use_container_width=True)
    
    if st.button("Go Back to Final Check"):
        st.session_state.page = 6
        st.rerun()


#############################################
# 6. PAGE NAVIGATION
#############################################

pages = {
    1: page1,
    2: page2,
    3: page3,
    4: page4,
    5: page5,
    6: page6,
    7: page7,
}

pages[st.session_state.page]()
