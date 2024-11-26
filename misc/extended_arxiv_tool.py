from modules.pdf_utils import process_pdf
from arxiv import Search, Client, SortCriterion
import matplotlib.pyplot as plt
import matplotlib
import requests

matplotlib.use('TkAgg')

target_id = '1706.03762'
query = 'id:' + target_id

client = Client()
search = Search(query=query, max_results=10, sort_by=SortCriterion.Relevance)

results = client.results(search)
parsed_pdf = None

for result in results:
    print(f'{result.entry_id} - {result.title} - {result.published} - {result.pdf_url}')

    if target_id in result.entry_id:
        file = requests.get(result.pdf_url).content
        parsed_pdf = process_pdf(stream=file)
        print('Got target PDF')

if parsed_pdf is not None:
    print(parsed_pdf['text'])

    images = parsed_pdf['images']
    nrows = len(images) // 2 + len(images) % 2
    fig, axes = plt.subplots(ncols=2, nrows=nrows, figsize=(15, 20))

    for i, img in enumerate(images):
        ax = axes[i // 2, i % 2]
        ax.imshow(img)
        ax.axis('off')

    plt.show()
