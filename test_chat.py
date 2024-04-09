# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire
import os

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

    Answer like [col1,col2,...]
    """}

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

    tabledir = "K3/"
    filenames = os.listdir(tabledir)
    starter = outputs["content"] + "Given the following relational table:\n"
    real_cols = []
    pred_cols = []
    i = 0
    for filename in filenames[0:10]:
        dialogs: List[Dialog] = []
        with open(tabledir + filename) as f:
            linelist = f.readlines()
            if len(linelist) > 20:
                f.close()
                continue
            colnames = linelist[0][:-1].split(',')
            real_cols.append(colnames)
            #print(f'Column Names: {colnames}')
            input = starter
            listcol = []
            for colnum in range(len(colnames)):
                listcol.append(f'col{colnum+1}')
            input += ';'.join(listcol) + '\n'
            lines = ''.join(linelist[1:])#.replace(',',';')
            input += lines + f'Guess the column names for the whole table. There are only {len(colnames)} columns in the table.'
            dialogs.append([{"role":"user", "content":input}]) #be sure to add outputs
        i += 1
        if i > max_batch_size - 1:
            break
        #break

        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            #print(result['generation']['content'].split('[')[1][:-1].split(', '))
            #pred_cols.append(result['generation']['content'].split('[')[1].split(']')[0].split(', '))
            print("\n==================================\n")

    correct = 0
    total = 0
    #for real,pred in zip(real_cols, pred_cols):
    #    for r,p in zip(real,pred):
    #        total += 1
    #        if r == p:
    #            correct += 1
    #print(f'Accuracy: {correct/total}')


if __name__ == "__main__":
    fire.Fire(main)

