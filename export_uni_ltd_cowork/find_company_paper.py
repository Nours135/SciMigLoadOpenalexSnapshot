import json
import os
import sys
from typing import Generator, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pickle

def load_one_dump_file(file_path):
    """
    Load a single dump file and return its content.
    
    Args:
        file_path (str): The path to the openalex dump file.
        
    Returns:
        dict: The content of the dump file.
    """
    data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f: 
            cur_d = json.loads(line.strip())
            # data.append(cur_d)
            yield cur_d
    #         import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    # return data


def dump_fpath_generator(folder):
    '''
    load all openalex dump files in a folder.
    Args: folder path
    '''
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.txt'):
                yield os.path.join(root, file)




# json path to aimed field


# trial 1, it seems openalex provides a type field of institution, now explore the value distribution of this field
# step 2, export examples of every type

# facility 研究所，算高校吧

def temp_explore_institution_type_filed_single(file_path):
    '''
    Explore the distribution of institution type in openalex dump files.
    Args: folder path
    '''
    type_dict = {'no_ins': 0, 'empty_ins': 0}
    example_dict = dict()  # to store examples of each type
    for item in load_one_dump_file(file_path):
        # import pdb; pdb.set_trace()
        for author_d in item.get('authorships', []):
            if 'institutions' in author_d:    # otherwise there is no institution information
                for institution in author_d['institutions']:
                    if 'type' in institution and institution['type'] is not None and len(institution['type']) > 0: 
                        type_value = institution['type']
                        if type_value not in type_dict:
                            type_dict[type_value] = 0
                            example_dict[type_value] = []
                        type_dict[type_value] += 1
                        if len(example_dict[type_value]) < 5:  # limit the number of examples to 5 for each type
                            example_dict[type_value].append(item)  # store the example item
                    else:
                        type_dict['empty_ins'] += 1
            else:
                type_dict['no_ins'] += 1

    return type_dict, example_dict

def temp_explore_institution_type_filed(folder):
    total_d = dict()
    file_paths = [item for item in dump_fpath_generator(folder)]
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(temp_explore_institution_type_filed_single, arg) for arg in file_paths[145:150]]   # DEBUG: only load some files for testing
        results = [future.result() for future in tqdm(futures, desc="Processing files")]
    
    exmaple_dicts = dict()
    for result in results:
        total_d = merge_value_dict(total_d, result[0])
        for key in set(list(result[1].keys()) + list(exmaple_dicts.keys())):
            if key not in exmaple_dicts:
                exmaple_dicts[key] = []
            if key in result[1]:
                exmaple_dicts[key].extend(result[1][key])
            # if len(exmaple_dicts[key]) > 5:  # limit the number of examples to 5 for each type
            #     exmaple_dicts[key] = exmaple_dicts[key][-5:]


    print(json.dumps(total_d, indent=4))
    with open('type_exmaple_dicts.pkl', 'wb') as f:
        pickle.dump(exmaple_dicts, f)

    return total_d


def merge_value_dict(dic1: Dict[Any, int], dic2: Dict[Any, int]) -> Dict[Any, int]:
    """
    Merge two dictionaries by summing the values of common keys.
    
    Args:
        dic1 (dict): The first dictionary.
        dic2 (dict): The second dictionary.
        
    Returns:
        dict: A new dictionary with merged values.
    """
    merged_dict = dic1.copy()
    
    for key, value in dic2.items():
        if key in merged_dict:
            merged_dict[key] += value
        else:
            merged_dict[key] = value
            
    return merged_dict




def is_education(author_d):
    '''
    Check if the author_d is an education institution.
    '''
    if 'institutions' in author_d and len(author_d['institutions']) > 0:
        institution = author_d['institutions'][0]
        if 'country_code' in institution and institution['country_code'] is not None and len(institution['country_code']) > 0:
            if institution['country_code'] != 'CN':
                return False  # only check Chinese Universities
        if 'type' in institution and institution['type'] is not None and len(institution['type']) > 0:
            if institution['type'] == 'education' or institution['type'] == 'facility':  # facility is also a type of education institution
                return True
            elif institution['type'] in ['other', 'nonprofit', 'archive', 'government', 'healthcare']:
                if 'raw_affiliation_string' in institution and institution['raw_affiliation_string'] is not None and len(institution['raw_affiliation_string']) > 0:
                    lower_raw_affiliation = institution['raw_affiliation_string'].lower()
                    keywords = ['university', 'college', 'institute', 'school', 'academy', 'faculty', 'department', 'research center', 'institute of technology']
                    for kw in keywords:
                        if kw in lower_raw_affiliation:
                            return True
    return False


def is_company(author_d):
    '''
    Check if the author_d is a Chinese company.
    '''
    def is_company_single_affiliation(institution):
        if 'country_code' in institution and institution['country_code'] is not None and len(institution['country_code']) > 0:
            if institution['country_code'] != 'CN':
                return False  # only check Chinese companies
        if 'type' in institution and institution['type'] is not None and len(institution['type']) > 0:
            if institution['type'] == 'company':  
                return True
            elif institution['type'] in ['other', 'archive', 'healthcare']:
                if 'raw_affiliation_string' in institution and institution['raw_affiliation_string'] is not None and len(institution['raw_affiliation_string']) > 0:
                    lower_raw_affiliation = institution['raw_affiliation_string'].lower()
                    keywords = ['ltd', 'limited', 'inc.', 'corporation', 'corp.', 'company', 'enterprise', 'co.', 'llc', 'plc', 'gmbh', 'ag', 'sarl']
                    for kw in keywords:
                        if kw in lower_raw_affiliation:
                            return True
        return False
    
    # consider all affiliations, if one of them is company, return True
    # if 'institutions' in author_d and len(author_d['institutions']) > 0:
    #     res = False
    #     for item in author_d['institutions']:
    #         if is_company_single_affiliation(item):
    #             res = True   # 或 
    #     return res

    if 'institutions' in author_d and len(author_d['institutions']) > 0:
        return is_company_single_affiliation(author_d['institutions'][0])   # only check the first affiliation
    return False

def is_CN(author_d):
    '''
    Check if the author_d is from China.
    '''
    if 'institutions' in author_d and len(author_d['institutions']) > 0:
        institution = author_d['institutions'][0]
        if 'country_code' in institution and institution['country_code'] is not None and len(institution['country_code']) > 0:
            if institution['country_code'] == 'CN':
                return True
    return False

# MAIN extract results

def single_extract_one_file(file_path):
    item_l = []
    cn_c = 0
    for item in load_one_dump_file(file_path):  # for each paper
        # import pdb; pdb.set_trace()
        first_author_company = False
        include_university = False
        for author_d in item.get('authorships', []):
            if author_d['author_position'] == 'first':
                if is_CN(author_d):  # only process Chinese papers
                    cn_c += 1
                else:
                    continue
                if is_company(author_d):
                    first_author_company = True
                else:
                    continue    # first author is not a company, skip this item
            elif not include_university and is_education(author_d):   # if include_university is true, skip this check, make process faster
                include_university = True

        if first_author_company and include_university:
            item_l.append(item)
                
    return {'item_l': item_l, 'CN_count': cn_c}

def main_extract_company_papers(folder):
    file_paths = [item for item in dump_fpath_generator(folder)]  #[150:156]   # DEBUG 50 files for testing
    output_file = 'company_papers.jsonl'
    out_fp = open(output_file, 'w', encoding='utf-8')
    chinese_paper_count = 0

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(single_extract_one_file, arg): arg for arg in file_paths}
        
        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(futures), total=len(file_paths), desc="Processing files"):
            try:
                result = future.result()  # 获取单个任务结果
                for item in result['item_l']:
                    out_fp.write(json.dumps(item, ensure_ascii=False) + '\n')
                chinese_paper_count += result['CN_count']
               
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    print(f"Total Chinese papers with first author as company and including university: {chinese_paper_count}")
    
   

if __name__ == "__main__":
    # step 1, explore json data structure
    # all_paths = [item for item in dump_fpath_generator('../works')]
    # temp_explore_institution_type_filed_single(all_paths[150])
    
    # data = load_one_dump_file(all_paths[200])
    # print(data) 


    # step 2, explore institution type field value distribution
    # total_d = temp_explore_institution_type_filed('../works')
    # import pdb; pdb.set_trace()


    # step 3, extract company papers
    main_extract_company_papers('../works')