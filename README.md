# Scientific Paper Explainer Agent

I'm attempting on qwen2.5-based agent creation. It's sole purpose is to explain scientific papers to me and my 
younger colleagues: sometimes it's difficult to understand complex method described in less than a dozen pages. So I'll
try to make my agent use full paper text, it's references, source codes (if there's any) and Tavily search engine 
to account as much relevant information as possible.

# Game plan

1. ~~Find sufficient chat model (tried: llama-3.1, llama-3.2vision, qwen2.5)~~
2. ~~Find or create an extended tool for arxiv search (I'd like to use paper PDFs to extract more information)~~
3. Make use of Tavily search engine for paper githubs (code also might be helpful)
4. ~~Make QnA tool for paper full texts and code sources to reduce context needs~~ HEEDS TESTING
5. Make web interface for agent with streamlit
6. ???
7. PROFIT