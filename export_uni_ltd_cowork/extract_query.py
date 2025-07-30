# 需要我从pdf文件中提取到

import pdfplumber
from tqdm import tqdm

def is_digit(s):
    """检查字符串是否为数字"""
    return s.isdigit() or (s.startswith('-') and s[1:].isdigit())

def is_letter(s):
    """检查字符串是否为字母"""
    return s.isalpha() or (s.startswith('-') and s[1:].isalpha())

# 提取前两列的内容，作为分类
current_letter = ''

cummulated_descriptions = []   # in order to add this, I have to append new rows after the paser meet the next economic code
last_code = ''
last_name = ''

extracted_data = []
with pdfplumber.open("domain.pdf") as pdf:
    for page in tqdm(pdf.pages):
        tables = page.extract_tables()
        for table in tables:
            # import pdb; pdb.set_trace()
            for row in table:
                # 提取前两个列
                if row[0] and row[1] and '行业代码' not in row[0]:
                    if row[0].isalpha(): current_letter = row[0]
                    elif is_digit(row[0]) and len(row[0]) == 2:
                        extracted_data.append((last_code, last_name, '\n'.join(set(cummulated_descriptions))))  # append last
                        # reset the cummulated descriptions
                        last_code = current_letter + row[0].strip()  # 可能是字母开头的代码
                        last_name = row[1].strip()
                        cummulated_descriptions = []
                        
                if row[3]:
                    cummulated_descriptions.append(row[3])  # 累积描述信息

# after all, append the last accumulated description
extracted_data.append((last_code, last_name, '\n'.join(set(cummulated_descriptions))))  # append last

# import pdb; pdb.set_trace()
# 将提取的数据保存到文件
import pandas as pd
df = pd.DataFrame(extracted_data, columns=['Category', 'Name', 'Description'])
df.to_csv('extracted_categories.csv', index=False)
# 现在可以在 'extracted_categories.csv' 中查看提取的分类数据