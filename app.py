
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost

# --- Configuration ---
st.set_page_config(
    page_title="Career Path Predictor",
    page_icon="🎓",
    layout="wide"
)

# --- Course Data ---
# A dictionary mapping the predicted career to a free, beginner-friendly course.
course_data = {
    'AI_ML_Specialist': ('Fundamentals of AI & ML', 'https://www.skillsoft.com/course/fundamentals-of-ai-ml-introduction-to-artificial-intelligence-25a4415c-6493-4324-ba6b-7622eb140ea4'),
    'API_Specialist': ('NGINX Kick-Start', 'https://www.f5.com/go/event/asean-on-demand-learning'),
    'Application_Support_Engineer': ('Introduction to iOS Mobile Application Development', 'https://www.coursera.org/learn/ios-app-development-basics'),
    'Business_Analyst': ('Business Analysis & Process Management', 'https://www.coursera.org/projects/business-analysis-process-management'),
    'Business_Intelligence_Analyst': ('IBM Business Intelligence (BI) Analyst Professional Certificate', 'https://www.coursera.org/professional-certificates/ibm-business-intelligence-analyst'),
    'Business_Systems_Analyst': ('Google Data Analytics Certificate', 'https://www.coursera.org/professional-certificates/google-data-analytics'),
    'CRM_Business_Analyst': ('CRM Business Analyst (Beginner)', 'https://lacroixsoft.odoo.com/slides/crm-business-analyst-beginner-81'),
    'Customer_Service_Executive': ('Becoming a Customer Service Executive', 'https://alison.com/course/becoming-a-customer-service-executive'),
    'Cyber_Security_Specialist': ('Cyber Security Course for Beginners - Level 01', 'https://www.udemy.com/course/certified-secure-netizen/'),
    'Data_Scientist': ('Python for Data Science', 'https://www.mygreatlearning.com/academy/learn-for-free/courses/python-for-data-science'),
    'Database_Administrator': ('Introduction to Database and SQL', 'https://www.mygreatlearning.com/academy/learn-for-free/courses/introduction-to-database-and-sql'),
    'Database_Developer': ('Introduction to Database and SQL', 'https://www.mygreatlearning.com/academy/learn-for-free/courses/introduction-to-database-and-sql'),
    'Database_Professional': ('Introduction to Database and SQL', 'https://www.mygreatlearning.com/academy/learn-for-free/courses/introduction-to-database-and-sql'),
    'Graphics_Designer': ('Get Started in Motion Graphics', 'https://www.nobledesktop.com/blog/get-started-in-motion-graphics-free-online-course'),
    'Hardware_Engineer': ('Introduction to Hardware and Operating Systems', 'https://www.coursera.org/learn/introduction-hardware-operating-systems'),
    'Helpdesk_Engineer': ('Google IT Support Certificate', 'https://www.coursera.org/professional-certificates/google-it-support'),
    'Information_Security_Analyst': ('Google Cybersecurity Professional Certificate', 'https://www.coursera.org/professional-certificates/google-cybersecurity'),
    'Information_Security_Specialist': ('Introduction to Cybersecurity', 'https://www.edx.org/course/introduction-to-cybersecurity'),
    'IT_Manager': ('Google Project Management Certificate', 'https://www.coursera.org/professional-certificates/google-project-management'),
    'Mobile_App_Developer': ('Meta Android Developer Professional Certificate', 'https://www.coursera.org/professional-certificates/meta-android-developer'),
    'Networking_Engineer': ('Introduction to Network Security', 'https://www.open.edu/openlearn/science-maths-technology/introduction-network-security/content-section-overview'),
    'Product_Manager': ('Google Project Management Certificate', 'https://www.coursera.org/professional-certificates/google-project-management'),
    'Programmer_Analyst': ('Python for Everybody', 'https://www.freecodecamp.org/learn/scientific-computing-with-python/'),
    'QA_Engineer': ('Introduction to Software Testing', 'https://www.coursera.org/learn/introduction-software-testing'),
    'Software_Developer': ('JavaScript Algorithms and Data Structures', 'https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/'),
    'Systems_Analyst': ('Google IT Support Certificate', 'https://www.coursera.org/professional-certificates/google-it-support'),
    'Technical_Support_Engineer': ('Google IT Support Certificate', 'https://www.coursera.org/professional-certificates/google-it-support'),
    'Technical_Writer': ('Google Technical Writing Fundamentals', 'https://developers.google.com/tech-writing/one'),
    'UX_UI_Designer': ('Google UX Design Certificate', 'https://www.coursera.org/professional-certificates/google-ux-design'),
    'Web_Developer': ('Responsive Web Design', 'https://www.freecodecamp.org/learn/responsive-web-design/'),
}


# --- Load Model and Components ---
# Use st.cache_resource to load these only once and speed up the app
@st.cache_resource
def load_model():
    try:
        model = joblib.load('xgboost_career_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        return model, preprocessor, label_encoder
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'xgboost_career_model.pkl', 'preprocessor.pkl', and 'label_encoder.pkl' are in the same directory.")
        return None, None, None

model, preprocessor, label_encoder = load_model()

# --- App Layout ---
st.title("🎓 Career Path Predictor")
st.markdown("Answer the following questions to get a personalized career path recommendation based on your skills and interests.")

if model is not None:
    with st.form("career_form"):
        st.header("Personal and Work Style")
        col1, col2, col3 = st.columns(3)
        with col1:
            hours_working_per_day = st.slider("How many hours per day do you prefer to work?", 4, 12, 8)
            introvert = st.radio("Are you more of an introvert?", ('Yes', 'No'), index=1)
        with col2:
            self_learning_capability = st.radio("Are you a strong self-learner?", ('Yes', 'No'), index=0)
            worked_in_teams = st.radio("Have you worked in teams before?", ('Yes', 'No'), index=0)
        with col3:
            can_work_long_time = st.radio("Can you work for long hours in front of a system?", ('Yes', 'No'), index=0)

        st.header("Academic and Technical Skills")
        st.markdown("Rate your skills and academic performance (0-100 for percentages, 0-10 for ratings).")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            os_skills = st.slider("Operating Systems (%)", 0, 100, 50)
            algo_skills = st.slider("Algorithms (%)", 0, 100, 50)
            prog_concepts_skills = st.slider("Programming Concepts (%)", 0, 100, 50)
            se_skills = st.slider("Software Engineering (%)", 0, 100, 50)
            cn_skills = st.slider("Computer Networks (%)", 0, 100, 50)

        with col_b:
            electronics_skills = st.slider("Electronics Subjects (%)", 0, 100, 50)
            ca_skills = st.slider("Computer Architecture (%)", 0, 100, 50)
            math_skills = st.slider("Mathematics (%)", 0, 100, 50)
            comm_skills = st.slider("Communication Skills (%)", 0, 100, 50)
            gpa = st.slider("Overall Academic Performance / GPA (out of 4.0)", 0.0, 4.0, 3.0, 0.1)


        with col_c:
            logical_quotient = st.slider("Logical Reasoning Rating (1-10)", 1, 10, 5)
            coding_skills = st.slider("Coding Skills Rating (1-10)", 1, 10, 5)
            public_speaking = st.slider("Public Speaking Rating (1-10)", 1, 10, 5)
            hackathons = st.slider("Number of Hackathons Attended", 0, 10, 1)


        st.header("Interests and Certifications")
        col_d, col_e = st.columns(2)
        with col_d:
            certifications = st.selectbox("Which certification area are you most interested in?",
                                          ['r programming', 'machine learning', 'information security', 'python',
                                           'shell programming', 'hadoop', 'distro making', 'full stack', 'app development'])
            workshops = st.selectbox("Which workshop area have you attended/are interested in?",
                                     ['data science', 'database security', 'game development', 'hacking',
                                      'system designing', 'testing', 'cloud computing', 'web technologies'])
        with col_e:
            extra_courses_did = st.selectbox("Have you taken extra courses?", ('Yes', 'No'), index=1)
            olympiads = st.selectbox("Have you participated in Olympiads?", ('Yes', 'No'), index=1)
            talent_tests = st.selectbox("Have you taken talent tests?", ('Yes', 'No'), index=1)


        # --- Submit Button ---
        submitted = st.form_submit_button("✨ Predict My Career Path")

        if submitted:
            # --- Data Collection and Preparation ---
            # Create a dictionary from the user's input
            user_input = {
                'AI_ML_Skills': 0, 'Academic_Performance_Score': gpa, 'Algorithms_Skills': algo_skills,
                'Business_Analysis_Skills': 0, 'Coding_Skills_Rating': coding_skills, 'Communication_Skills': comm_skills,
                'Computer_Architecture_Skills': ca_skills, 'Computer_Forensics_Skills': 0, 'Cyber_Security_Skills': 0,
                'Data_Science_Skills': 0, 'Database_Skills': 0, 'Distributed_Computing_Skills': 0,
                'Electronics_Skills': electronics_skills, 'Extra-courses did': extra_courses_did.lower(),
                'Graphics_Designing_Skills': 0, 'Hours working per day': hours_working_per_day,
                'Introvert': 1 if introvert == 'Yes' else 0, 'Java_Skills': 0, 'Logical_Reasoning_Skills': logical_quotient,
                'Mathematics_Skills': math_skills, 'Memory_Capability_Score': 'medium', 'Networking_Skills': cn_skills,
                'Operating_Systems_Skills': os_skills, 'Programming_Skills': prog_concepts_skills,
                'Project_Management_Skills': 0, 'Public_Speaking_Skills': public_speaking, 'Python_Skills': 0,
                'Reading_Writing_Skills': 'medium', 'SQL_Skills': 0, 'Software_Development_Skills': 0,
                'Software_Engineering_Skills': se_skills, 'Technical_Communication_Skills': 0,
                'Troubleshooting_Skills': 0, 'can work long time before system?': 1 if can_work_long_time == 'Yes' else 0,
                'certifications': certifications, 'hackathons': hackathons, 'olympiads': olympiads.lower(),
                'self-learning capability?': 1 if self_learning_capability == 'Yes' else 0,
                'talenttests taken?': talent_tests.lower(), 'worked in teams ever?': 1 if worked_in_teams == 'Yes' else 0,
                'workshops': workshops
            }

            # Convert to a DataFrame with the correct column order
            input_df = pd.DataFrame([user_input])
            
            # --- Prediction ---
            # Preprocess the input data
            input_processed = preprocessor.transform(input_df)

            # Make a prediction
            prediction_encoded = model.predict(input_processed)

            # Decode the prediction
            predicted_career = label_encoder.inverse_transform(prediction_encoded)[0]

            # --- Display Results ---
            st.success(f"### Recommended Career Path: **{predicted_career.replace('_', ' ')}**")

            # Get and display the recommended course
            course_title, course_url = course_data.get(predicted_career, ("No course found", "#"))
            st.info(f"#### To get started, check out this free beginner's course:")
            st.markdown(f"###  курс: [{course_title}]({course_url})")

