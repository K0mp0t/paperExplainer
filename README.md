# Scientific Paper Explainer Agent

I'm attempting on llama3.2-vision based agent creation. It's sole purpose is to explain scientific papers to me and my 
younger colleagues: sometimes it's difficult to understand complex method described in 8 (sometimes less) pages. So I'll
try to make my agent use both text and images from paper PDFs, it's references, source codes (if there's any) and 
Tavily search engine to account as much relevant information as possible.

# Game plan

1. ~~Make Llama3.2-vision model be able to call tools~~ (it couldn't)
2. Find or create an extended tool for arxiv search (I'd like to use paper PDFs to extract more information)
3. Make use of Tavily search engine for paper githubs (code also might be helpful)
4. ???
5. PROFIT