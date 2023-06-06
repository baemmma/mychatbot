
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat

st.set_page_config(page_title="Vivabot - An LLM-powered Streamlit app")

with open("chatbot_database.txt", "r") as file:
    chatbot_db = file.read()

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 HugChat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [HugChat](https://github.com/Soulter/hugging-chat-api)
    - [OpenAssistant/oasst-sft-6-llama-30b-xor](https://huggingface.co/OpenAssistant/oasst-sft-6-llama-30b-xor) LLM model
    
    💡 Note: No API key required!
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Data Professor](https://youtube.com/dataprofessor)')

# Generate empty lists for generated and past.
## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm HugChat, How may I help you?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

# User input
def greetings(sentence, greetings_inputs, greetings_outputs):
    if sentence.lower() in [x.lower() for x in greetings_inputs]:
        output = random.choice(greetings_outputs)
        return output
    else:
        return None

def get_closest_sentence(query, tf_idf, vectorizer, database):
    query_tf_idf = vectorizer.transform([query])
    similarity = cosine_similarity(query_tf_idf, tf_idf)
    closest_sentence_index = np.argmax(similarity)
    return database[closest_sentence_index]

def vivabot(greetings_inputs, greetings_outputs, tf_idf, vectorizer, database):
    print("Welcome to Vivabot!")
    print("How can I assist you today?")
    
    while True:
        user_input = input("User input: ")
        
        if user_input.lower() == "bye":
            print("Goodbye! Have a great day!")
            break
        
        greetings_output = greetings(user_input, greetings_inputs, greetings_outputs)
        
        if greetings_output:
            print(greetings_output)
        else:
            closest_sentence = get_closest_sentence(user_input, tf_idf, vectorizer, database)
            print(closest_sentence)


## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
