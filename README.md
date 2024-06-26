# 🦙📚 LlamaIndex - Chat with the Streamlit docs

Build a chatbot powered by LlamaIndex that augments GPT 3.5 with the content of the Streamlit docs (or your own data).

## Overview of the App

<img src="app.png" width="75%">

- Takes user queries via Streamlit's `st.chat_input` and displays both user queries and model responses with `st.chat_message`
- Uses LlamaIndex to load and index data and create a chat engine that will retrieve context from that data to respond to each user query

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lllamaindex-chat-with-docs.streamlit.app/)

## Get an OpenAI API key

You can get your own OpenAI API key by following the following instructions:
1. Go to https://platform.openai.com/account/api-keys.
2. Click on the `+ Create new secret key` button.
3. Next, enter an identifier name (optional) and click on the `Create secret key` button.

## Try out the app

Once the app is loaded, enter your question about the Streamlit library and wait for a response.
#   C h a t B o t 
 
 Here is a presentation slide based on the provided information for "Wrapped Spatial Gaussian Process":

---

## Wrapped Spatial Gaussian Process

### Gaussian Process Definition
- **Gaussian Process (GP)**: Denoted as \( Y(s) \)
  - **Input**: \( s \in \mathbb{R}^2 \)
  - **Mean Function**: \( \mu(s) \)
  - **Covariance Function**: \( \Sigma^2(\|s_i - s_j\|) \)
- **Covariance Function (\(\Sigma^2\))**: 
  - Provides correlation between variables at locations \( s_i \) and \( s_j \)
  - Indexed by parameters \( \theta \)
  - Homogeneous spatial variance

### Multivariate Normal Distribution
- For locations \( s_1, s_2, \ldots, s_n \):
  \[
  Y = (Y(s_1), Y(s_2), \ldots, Y(s_n)) \sim \mathcal{N}(\mu, \Sigma^2 R(\cdot))
  \]
  - **Mean Vector (\(\mu\))**:
    \[
    \mu = (\mu(s_1), \mu(s_2), \ldots, \mu(s_n))
    \]
  - **Correlation Matrix (\(R(\cdot)\))**:
    \[
    R(\cdot)_{ij} = \Sigma^2(\|s_i - s_j\|)
    \]

### Wrapped Normal Distribution
- **Wrapped Random Process**:
  \[
  \Theta \sim \text{WrapN}(\mu, \Sigma^2 R(\cdot))
  \]
  - **WrapN(·, ·)**: Indicates the wrapped normal distribution

### Key Points
- The wrapped spatial GP extends the traditional GP to model spatial data with circular or angular components.
- **Applications**: Often used in fields like environmental statistics, geostatistics, and spatial analysis where data exhibits periodic behavior.
- **Importance**: Captures spatial correlations effectively, allowing for better predictions and uncertainty quantifications in spatial models.

---

This slide provides a clear and concise explanation of the wrapped spatial Gaussian process, its components, and its significance.
