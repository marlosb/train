import datetime

from abt import create_abt
from train import train_select_model
   
DATA_PATH = 'data/olist_dsa.db'
QUERY_PATH = 'sql/Script_ABT_olist_dtref_safra_20200818.sql'

primeira_safra = "2018-03-01"
ultima_safra = "2018-06-01"

if __name__ == "__main__":
    create_abt(QUERY_PATH, DATA_PATH, primeira_safra, ultima_safra)
    train_select_model(DATA_PATH)