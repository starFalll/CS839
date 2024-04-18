# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire
import numpy as np
import os
from json import loads
import pandas as pd
import re
from tqdm import tqdm

from llama import Llama, Dialog


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 4096, #originally 512
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    outputs = {"role" : "system", "content" : """
    Column Names are limited to the following:
    name, description, team, type, age, location, year, city, rank, status, state, category,
    weight, code, club, artist, result, position, country, notes, class, company, album, symbol,
    address, duration, format, county, day, gender, industry, language, sex, product, jockey,
    region, area, service, teamName, order, isbn, fileSize, grades, publisher, plays, origin,
    elevation, affiliation, component, owner, genre,  manufacturer, brand, family, credit, depth,
    classification, collection, species, command, nationality, currency, range, affiliate,
    birthDate, ranking, capacity, birthPlace, person, creator, operator, religion, education,
    requirement, director, sales, continent, organisation
    Do not use any column names aside from these.
    
    Output must be in valid JSON like the following example {"column" : "col1"}. Give only 1 prediction. Do NOT add any explanation beyond the JSON.
    """} #note: teamName, fileSize, birthDate, and birthPlace contain uppercase

    #dialogs: List[Dialog] = [
        #[outputs, {"role" : "user", "content": """
        #Given the following relational table:
        #Florence Nightingale;1820-05-12;Nurse;Florence
        #Marie Curie;1867-11-07;Chemist; Warsaw
        #Alan Turing;1912-06-23;Comp Scientist;London
        #Johann Gauss;1777-04-30;Mathematician;Braunschweig
        #Guess the column names for the whole table
        #"""}],
    #]

    tabledir = "K4"
    filenames = os.listdir(tabledir + '/')
    starter = "Given the following column values in a relational table: "
    real_cols = []
    pred_cols = []
    for filename in tqdm(filenames[0:100]):
        dialogs: List[Dialog] = []
        fullpath = tabledir + '/' + filename
        with open(fullpath) as f:
            real_cols += f.readline()[:-1].split(',')
        df = pd.read_csv(fullpath).astype(str)
        for col in df.columns:
            input = starter
            #real_cols.append(col)
            input += ", ".join(df[col][0:20])
            input += '\nGuess the column name'
            dialogs.append([outputs, {"role":"user", "content":input}])
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        #tmpcols = []
        for dialog, result in zip(dialogs, results):
            #for msg in dialog:
            #    print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            #print(
            #    f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            #)
            #pred_cols.append(result['generation']['content'].replace('\n',' ').split('.')[0].split()[-1])
            pred_cols.append(loads(result['generation']['content'])["column"])
            #print(pred_cols[-1])
            #print("\n==================================\n")
        #pred_cols.append(tmpcols)

    with open(tabledir + '_true.npy', 'wb') as f:
        np.save(f, np.array(real_cols, dtype='<U14'))
    with open(tabledir + '_pred.npy', 'wb') as f:
        np.save(f, np.array(pred_cols, dtype='<U14'))

    """correct = 0
    total = 0
    for real,pred in zip(real_cols, pred_cols):
        for r,p in zip(real,pred):
            total += 1
            if r == (p[0].lower() + p[1:]):
                correct += 1
    print(f'Accuracy: {correct/total}')"""


if __name__ == "__main__":
    fire.Fire(main)
