import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import time
import json
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Recommender Engine 4.1",
    page_icon="🚀",
    layout="wide"
)

# --- Styling ---
st.markdown("""
<style>
    .stCodeBlock code {
        font-family: 'Courier New', Courier, monospace; font-size: 1.1em;
    }
    h2 {
        color: #FAFAFA; border-bottom: 2px solid #333; padding-bottom: 5px;
    }
    .st-emotion-cache-16txtl3 {
        padding-top: 2rem;
    }
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper & Animation Functions ---
def interpolate_vector(start_vec, end_vec, alpha):
    return start_vec * (1 - alpha) + end_vec * alpha

def interpolate_color(start_color, end_color, alpha):
    start_rgb = np.array([int(start_color[i:i+2], 16) for i in (1, 3, 5)])
    end_rgb = np.array([int(end_color[i:i+2], 16) for i in (1, 3, 5)])
    inter_rgb = start_rgb * (1 - alpha) + end_rgb * alpha
    return '#{:02x}{:02x}{:02x}'.format(int(inter_rgb[0]), int(inter_rgb[1]), int(inter_rgb[2]))

def generate_vector(value, seed_base=42, size=8):
    valid_seed = int(abs(seed_base * value)) % (2**32)
    np.random.seed(valid_seed)
    return np.random.randint(0, 100, size=size)

def create_nn_graph(activations, selected_node=None):
    dot = graphviz.Digraph(engine='dot')
    dot.attr('graph', rankdir='LR', splines='spline', bgcolor='transparent', compound='true')
    dot.attr('node', shape='circle', style='filled', fontcolor='#FAFAFA', color='#999999', penwidth='0.5')
    dot.attr('edge', color='#666666', arrowhead='none', penwidth='0.5')

    input_nodes = {'Demographics': 'blue', 'Transactions': 'orange', 'Behavioral': 'green', 'External': 'purple'}
    processing_layer_1 = {f'P1_{i}': 'gray' for i in range(5)}
    processing_layer_2 = {f'P2_{i}': 'gray' for i in range(5)}
    persona_nodes = {'Thrifty Homeowner': 'yellow', 'Young Professional': 'cyan', 'Global Traveler': 'magenta'}
    output_nodes = {'Personal Loan': 'red', 'Wealth Management': 'red', 'Travel Rewards Card': 'red'}
    all_nodes = [input_nodes, processing_layer_1, processing_layer_2, persona_nodes, output_nodes]

    for i, layer in enumerate(all_nodes):
        with dot.subgraph(name=f'cluster_{i}') as c:
            c.attr(style='invis')
            for node_name in layer:
                activation = activations.get(node_name, 0.1)
                fill_color = interpolate_color("#222222", "#FFFFFF", activation)
                font_color = "#111111" if activation > 0.6 else "#FAFAFA"
                
                if node_name == selected_node:
                    c.node(node_name, label='', fillcolor=fill_color, fontcolor=font_color,
                           penwidth='3.0', color='#00FFFF', shadowcolor='#00FFFF', style='filled,bold,diagonals')
                else:
                    c.node(node_name, label='', fillcolor=fill_color, fontcolor=font_color,
                           penwidth=str(0.5 + activation * 2), color=interpolate_color("#666666", "#FFFFFF", activation))

    for i in range(len(all_nodes) - 1):
        for start_node in all_nodes[i]:
            for end_node in all_nodes[i+1]:
                start_act = activations.get(start_node, 0)
                end_act = activations.get(end_node, 0)
                path_act = (start_act + end_act) / 2
                if path_act > 0.1:
                    pen_width = str(0.2 + path_act * 1.5)
                    edge_color = interpolate_color("#333333", "#FFFFFF", path_act)
                    dot.edge(start_node, end_node, color=edge_color, penwidth=pen_width)
    return dot

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# --- Session State ---
if 'previous_input_dict' not in st.session_state:
    st.session_state.previous_input_dict = {}
    st.session_state.previous_input_json = ""
    st.session_state.previous_outputs = None

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("⚙️ Real-Time Client Inputs")
    st.markdown("### <span style='color:#3b82f6;'>👤 Demographics & History</span>", unsafe_allow_html=True)
    age = st.slider("Client Age", 18, 80, 35)
    loyalty_tier = st.select_slider("Loyalty Tier", ["Bronze", "Silver", "Gold"], value="Silver")
    st.markdown("### <span style='color:#f97316;'>💳 Transaction Data</span>", unsafe_allow_html=True)
    last_tx_category = st.selectbox("Last Transaction Category", ['Home Improvement', 'Luxury Goods', 'Travel'], index=0)
    last_tx_amount = st.slider("Last Transaction Amount ($)", 50, 5000, 450)
    st.markdown("### <span style='color:#22c55e;'>📈 Behavioral Patterns</span>", unsafe_allow_html=True)
    recent_inquiry = st.selectbox("Recent Product Inquiry", ["None", "Mortgage", "Personal Loan"], index=0)
    st.markdown("### <span style='color:#8b5cf6;'>🌐 External Data</span>", unsafe_allow_html=True)
    credit_score = st.slider("Credit Score", 300, 850, 720)

current_inputs = {'age': age, 'loyalty': loyalty_tier, 'tx_cat': last_tx_category, 'tx_amount': last_tx_amount, 'inquiry': recent_inquiry, 'credit': credit_score}
current_inputs_json = json.dumps(current_inputs, sort_keys=True)
inputs_changed = (st.session_state.previous_input_json != current_inputs_json)

# --- Core Logic Simulation ---
def get_model_outputs(inputs):
    activations = {node: 0.1 for node in ['Demographics', 'Transactions', 'Behavioral', 'External', *[f'P1_{i}' for i in range(5)], *[f'P2_{i}' for i in range(5)]]}
    scores = {"Personal Loan": 10, "Wealth Management": 10, "Travel Rewards Card": 10}
    persona = "Young Professional"
    recommendation = "Wealth Management"

    activations['Demographics'] = 0.8; activations['Transactions'] = 0.8; activations['External'] = 0.8
    if inputs.get('inquiry', 'None') != "None": activations['Behavioral'] = 0.8

    if inputs.get('inquiry') == "Personal Loan" or (inputs.get('tx_cat') == "Home Improvement" and inputs.get('credit', 0) > 680):
        persona = "Thrifty Homeowner"
        recommendation = "Personal Loan"
        activations.update({'Thrifty Homeowner': 0.95, 'P1_0': 0.8, 'P1_1': 0.7, 'P2_1': 0.9, 'P2_3': 0.6, 'Personal Loan': 0.9})
        scores["Personal Loan"] += 50
    elif inputs.get('tx_cat') == "Travel" and inputs.get('loyalty') == "Gold":
        persona = "Global Traveler"
        recommendation = "Travel Rewards Card"
        activations.update({'Global Traveler': 0.95, 'P1_2': 0.6, 'P1_4': 0.9, 'P2_0': 0.7, 'P2_4': 0.9, 'Travel Rewards Card': 0.9})
        scores["Travel Rewards Card"] += 60
    else:
        activations.update({'Young Professional': 0.95, 'P1_3': 0.9, 'P2_2': 0.8, 'Wealth Management': 0.9})
        scores["Wealth Management"] += 40

    probabilities = softmax(np.array(list(scores.values())))
    confidence = np.max(probabilities)
    shap_values = {k: v - 10 for k, v in scores.items()}

    return {"persona": persona, "recommendation": recommendation, "activations": activations, "shap_values": shap_values, "confidence": confidence}

# --- Main Dashboard ---
st.title("🚀 AI Recommender Engine")
st.markdown("An interactive visualization of a neural network for cross-selling financial products.")
st.markdown("---")

col_vecs, col_nn, col_xai = st.columns([1, 1.5, 1])
current_outputs = get_model_outputs(current_inputs)
if st.session_state.previous_outputs is None:
    st.session_state.previous_outputs = current_outputs
    st.session_state.previous_input_dict = current_inputs # Initialize on first run

vec_placeholder = col_vecs.empty()
nn_placeholder = col_nn.empty()
xai_placeholder = col_xai.empty()
previous_outputs = st.session_state.previous_outputs
previous_input_dict = st.session_state.previous_input_dict

with st.sidebar:
    st.markdown("---")
    st.header("🔍 AI Node Inspector")
    all_node_names = ['Demographics', 'Transactions', 'Behavioral', 'External', *[f'P1_{i}' for i in range(5)], *[f'P2_{i}' for i in range(5)], 'Thrifty Homeowner', 'Young Professional', 'Global Traveler', 'Personal Loan', 'Wealth Management', 'Travel Rewards Card']
    selected_node = st.selectbox("Select a node to inspect:", all_node_names, index=15)

# --- Animation or Static Display ---
animation_steps = 15 if inputs_changed else 0
for i in range(animation_steps + 1):
    # FIX: Prevents ZeroDivisionError when only the node inspector changes
    alpha = i / animation_steps if animation_steps > 0 else 1.0

    inter_activations = {
        node: interpolate_vector(np.array(previous_outputs['activations'].get(node, 0.1)), np.array(current_outputs['activations'].get(node, 0.1)), alpha)
        for node in current_outputs['activations']
    }

    with vec_placeholder.container():
        st.subheader("🔢 Data Vectors")
        # Generate all vectors with interpolation
        age_vec = interpolate_vector(generate_vector(previous_input_dict.get('age', 35)), generate_vector(age), alpha)
        tx_vec = interpolate_vector(generate_vector(previous_input_dict.get('tx_amount', 450)), generate_vector(last_tx_amount), alpha)
        # FIX: Added missing Behavioral and External vectors
        inquiry_num = len(recent_inquiry) # Simple way to make string into a number for vector generation
        prev_inquiry_num = len(previous_input_dict.get('inquiry', "None"))
        behavior_vec = interpolate_vector(generate_vector(prev_inquiry_num, seed_base=50), generate_vector(inquiry_num, seed_base=50), alpha)
        external_vec = interpolate_vector(generate_vector(previous_input_dict.get('credit', 720)), generate_vector(credit_score), alpha)

        # Display all vectors
        st.markdown("<p style='color:#3b82f6;'>Demographics</p>", unsafe_allow_html=True)
        st.code(f"[{' '.join(f'{x:02d}' for x in age_vec.astype(int))}]")
        st.markdown("<p style='color:#f97316;'>Transactions</p>", unsafe_allow_html=True)
        st.code(f"[{' '.join(f'{x:02d}' for x in tx_vec.astype(int))}]")
        st.markdown("<p style='color:#22c55e;'>Behavioral</p>", unsafe_allow_html=True)
        st.code(f"[{' '.join(f'{x:02d}' for x in behavior_vec.astype(int))}]")
        st.markdown("<p style='color:#8b5cf6;'>External Data</p>", unsafe_allow_html=True)
        st.code(f"[{' '.join(f'{x:02d}' for x in external_vec.astype(int))}]")

    with nn_placeholder.container():
        st.subheader("🧠 Neural Network Activation")
        st.graphviz_chart(create_nn_graph(inter_activations, selected_node), use_container_width=True)

    with xai_placeholder.container():
        st.subheader("🔬 Explainable AI (XAI)")
        final_alpha = 1.0 if i == animation_steps else alpha
        inter_persona = current_outputs['persona'] if final_alpha > 0.5 else previous_outputs['persona']
        inter_rec = current_outputs['recommendation'] if final_alpha > 0.5 else previous_outputs['recommendation']
        inter_conf = interpolate_vector(np.array(previous_outputs['confidence']), np.array(current_outputs['confidence']), alpha)

        with st.expander("Persona & Recommendation", expanded=True):
            st.metric(label="Discovered Client Persona", value=inter_persona)
            st.metric(label="✅ Top Recommendation", value=inter_rec, help=f"The model is {inter_conf:.0%} confident.")
            st.progress(inter_conf, text=f"Confidence: {inter_conf:.0%}")
        
        if i == animation_steps:
            with st.expander("Business Impact", expanded=True):
                success_rate = 0.75 if inter_persona == 'Thrifty Homeowner' else (0.90 if inter_persona == 'Global Traveler' else 0.60)
                potential_revenue = 1250 if inter_rec == 'Personal Loan' else (850 if inter_rec == 'Travel Rewards Card' else 2500)
                st.metric(label="Predicted Success Rate", value=f"{success_rate:.0%}")
                st.metric(label="Potential Revenue (Est.)", value=f"${potential_revenue:,}")

            with st.expander("Feature Importance (SHAP Simulation)"):
                shap_df = pd.DataFrame(list(current_outputs['shap_values'].items()), columns=['Feature', 'Impact']).set_index('Feature')
                fig, ax = plt.subplots(); shap_df['Impact'].plot(kind='barh', ax=ax, color=['#ef4444' if x < 0 else '#3b82f6' for x in shap_df['Impact']])
                ax.set_facecolor("#0e1117"); fig.set_facecolor("#0e1117"); ax.spines['bottom'].set_color('white'); ax.spines['left'].set_color('white')
                ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white'); ax.set_xlabel("Impact on Recommendation", color='white')
                st.pyplot(fig, use_container_width=True)

    if inputs_changed: time.sleep(0.01)

# --- Update Session State ---
if inputs_changed:
    st.session_state.previous_input_json = current_inputs_json
    st.session_state.previous_input_dict = current_inputs
    st.session_state.previous_outputs = current_outputs

# --- Node Inspector Logic and Display ---
def get_node_explanation(node_name, inputs, activations):
    explanation = {"reason": "This is a foundational input node.", "drivers": [], "impacts": []}
    if "P1" in node_name:
        explanation = {"reason": "This node combines multiple inputs into an abstract feature.", "drivers": ["Demographics", "Transactions", "Behavioral", "External"], "impacts": [f'P2_{i}' for i in range(5)]}
    elif "P2" in node_name:
        explanation = {"reason": "This node identifies more refined patterns from the previous layer.", "drivers": [f'P1_{i}' for i in range(5)], "impacts": ["Thrifty Homeowner", "Young Professional", "Global Traveler"]}
    elif node_name == "Thrifty Homeowner":
        drivers = []
        if inputs.get('inquiry') == "Personal Loan": drivers.append("Behavioral (Loan Inquiry)")
        if inputs.get('tx_cat') == "Home Improvement": drivers.append("Transactions (Home Spending)")
        if inputs.get('credit', 0) > 680: drivers.append("External (High Credit)")
        explanation = {"reason": "Activated by home-related spending, loan inquiries, and good credit.", "drivers": drivers or ["General Profile"], "impacts": ["Personal Loan"]}
    elif node_name == "Global Traveler":
        drivers = []
        if inputs.get('tx_cat') == "Travel": drivers.append("Transactions (Travel Spending)")
        if inputs.get('loyalty') == "Gold": drivers.append("Demographics (Gold Loyalty)")
        explanation = {"reason": "Activated by travel-related spending and high loyalty status.", "drivers": drivers or ["General Profile"], "impacts": ["Travel Rewards Card"]}
    elif node_name == "Young Professional":
        explanation = {"reason": "This is the default persona when specific patterns aren't met. It represents a generally good client profile.", "drivers": ["Demographics", "External"], "impacts": ["Wealth Management"]}
    return explanation

with st.sidebar:
    activation_level = current_outputs['activations'].get(selected_node, 0.1)
    st.progress(activation_level, text=f"Activation: {activation_level:.0%}")
    explanation = get_node_explanation(selected_node, current_inputs, current_outputs['activations'])
    st.markdown(f"**🧠 Reasoning:** *{explanation['reason']}*")
    if explanation['drivers']:
        st.markdown("**Upstream Drivers:**")
        st.write(", ".join(explanation['drivers']))
    if explanation['impacts']:
        st.markdown("**Downstream Impact:**")
        st.write(", ".join(explanation['impacts']))