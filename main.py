import datetime

from abt import create_abt
from train import train_select_model
   
DATA_PATH = 'data/olist_dsa.db'
QUERY_PATH = 'sql/Script_ABT_olist_dtref_safra_20200818.sql'

primeira_safra = "2018-03-01"
ultima_safra = "2018-06-01"

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    create_abt(QUERY_PATH, DATA_PATH, primeira_safra, ultima_safra)
    mid_time = datetime.datetime.now()
    train_select_model(DATA_PATH)
    end_time = datetime.datetime.now()

    print('==========================================================')
    print('\nScrip Completo')
    print(f'A ABT foi criada em: {mid_time - start_time}')
    print(f'Os modelos foram criados em {end_time - mid_time}')
    print(f'O tempo total foi { end_time - start_time}\n')
    print('==========================================================')