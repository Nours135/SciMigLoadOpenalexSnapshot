# 需要我从pdf文件中提取到

import pdfplumber
from tqdm import tqdm

# 提取前两列的内容，作为分类
extracted_data = []
with pdfplumber.open("domain.pdf") as pdf:
    for page in tqdm(pdf.pages):
        tables = page.extract_tables()
        for table in tables:
            # import pdb; pdb.set_trace()
            for row in table:
                # 提取前两个列
                if row[0] and row[1] and '行业代码' not in row[0]:
                    extracted_data.append((row[0], row[1]))


# import pdb; pdb.set_trace()
# 将提取的数据保存到文件
import pandas as pd
df = pd.DataFrame(extracted_data, columns=['Category', 'Description'])
df.to_csv('extracted_categories.csv', index=False)
# 现在可以在 'extracted_categories.csv' 中查看提取的分类数据