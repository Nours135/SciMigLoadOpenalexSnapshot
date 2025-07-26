# conver jsonl to csv

import pandas as pd
import json
from tqdm import tqdm


def jsonl_to_csv(jsonl_file, csv_file):
    fp = open(jsonl_file, 'r', encoding='utf-8')
    
    keys = ['id', 'doi', 'title', 'publication_date', 'concepts', 'keywords']

    df_data = []

    for line in tqdm(fp):
        cur_d = json.loads(line.strip())
        tmp = dict()
        for key in keys:
            if key in cur_d:
                tmp[key] = cur_d[key]
            else:
                tmp[key] = 'N/A'
        # process authorships
        authorships = cur_d.get('authorships', [])
        first_author = []
        other_authors = []
        for author in authorships:
            if author.get('author_position', '') == 'first':
                first_author.append(author)
            else:
                other_authors.append(author)

        tmp['first_author'] = json.dumps(first_author) if first_author else 'N/A'
        tmp['other_authors'] = json.dumps(other_authors) if other_authors else 'N/A'

        df_data.append(tmp)
    
    df = pd.DataFrame(df_data, columns=keys + ['first_author', 'other_authors'])
    df.to_excel(csv_file, index=False)



if __name__ == "__main__":
    jsonl_file = 'company_papers.jsonl'
    csv_file = 'company_papers.xlsx'
    
    # Convert JSONL to CSV
    jsonl_to_csv(jsonl_file, csv_file)
    
    