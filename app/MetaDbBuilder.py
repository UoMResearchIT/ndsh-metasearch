import datetime
import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from cleantext import clean

class MetaData:
    def __init__(self, ):
        self.recordId = None
        self.identifier = None
        self.title = None
        self.abstract = None
        self.hierarchy_level = None
    
    def to_dict(self):
        return {
            'identifier': self.identifier, 
            'recordId': self.recordId,
            'title': self.title, 
            'abstract': self.abstract,
            'hierarchy_level': self.hierarchy_level,
            }

class TextCleaner:
    @classmethod
    def default_clean(cls, text: str) -> str:
        if not text:
            return text
        return clean(text,
        fix_unicode=True,               # fix various unicode errors
        to_ascii=True,                  # transliterate to closest ASCII representation
        lower=True,                     # lowercase text
        no_line_breaks=True,            # fully strip line breaks as opposed to only normalizing them
        no_urls=False,                  # replace all URLs with a special token
        no_emails=False,                # replace all email addresses with a special token
        no_phone_numbers=False,         # replace all phone numbers with a special token
        no_numbers=False,               # replace all numbers with a special token
        no_digits=False,                # replace all digits with a special token
        no_currency_symbols=False,      # replace all currency symbols with a special token
        no_punct=False,                 # remove punctuations
        #replace_with_punct="",         # instead of removing punctuations you may replace them
        #replace_with_url="<URL>",
        #replace_with_email="<EMAIL>",
        #replace_with_phone_number="<PHONE>",
        #replace_with_number="<NUMBER>",
        #replace_with_digit="0",
        #replace_with_currency_symbol="<CUR>",
        lang="en"                       # set to 'de' for German special handling
        )
    
    @staticmethod
    def clean_column(df: pd.DataFrame, col: str, clean_func:callable) -> pd.DataFrame:
        df[f"raw_{col}"] = df[col]
        df[col] = df[col].apply(clean_func)
        return df

class XmlParser:
    @classmethod
    def ceda(cls, xml_fp: str) -> list[MetaData]:
        namespaces = {
            'csw': 'http://www.opengis.net/cat/csw/2.0.2',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'dct': 'http://purl.org/dc/terms/'
        }
        
        
        tree = ET.parse(xml_fp)
        root = tree.find('csw:SearchResults', namespaces=namespaces)
        records = root.findall('csw:Record', namespaces=namespaces)
        mds = []
        for record in records:
            md = MetaData()
            md.identifier = record.find('dc:identifier', namespaces=namespaces).text
            md.title = record.find('dc:title', namespaces=namespaces).text
            md.abstract = record.find('dct:abstract', namespaces=namespaces).text
            mds += [md]
        return mds
    
    @classmethod
    def gmd(cls, xml_fp: str) -> list[MetaData]:

        def find_text(node: ET.Element, xpath: str, namespaces: dict) -> str:
            '''return the text of the first node found by xpath. None if not found'''
            v = node.find(xpath, namespaces=namespaces)
            return v.text if v is not None else None
            

        namespaces = {
            'csw': "http://www.opengis.net/cat/csw/2.0.2",
            'gmd': "http://www.isotc211.org/2005/gmd",
            'gco': "http://www.isotc211.org/2005/gco",
            'srv': "http://www.isotc211.org/2005/srv"
        }

        tree = ET.parse(xml_fp)
        root = tree.find('csw:SearchResults', namespaces=namespaces)
        records = root.findall('gmd:MD_Metadata', namespaces=namespaces)
        mds = []
        _recordId = int(os.path.basename(xml_fp).replace('.xml', '').replace('nerc_', ''))
        
        for record in records:
            data_identification_info_nodes = record.findall("gmd:identificationInfo", namespaces=namespaces)
            data_identification_nodes = record.findall("gmd:identificationInfo/gmd:MD_DataIdentification", namespaces=namespaces)
            service_identification_nodes = record.findall('gmd:identificationInfo/srv:SV_ServiceIdentification', namespaces=namespaces)
            
            if len(data_identification_info_nodes) > 1:
                raise ValueError('More than one identificationInfo node found.')
            if len(data_identification_nodes) > 1:
                raise ValueError('More than one MD_DataIdentification node found.')
            if len(service_identification_nodes) > 1:
                raise ValueError('More than one SV_ServiceIdentification node found.')

            
            md = MetaData()
            md.recordId = _recordId
            
            md.identifier = find_text(record, 'gmd:fileIdentifier/gco:CharacterString', namespaces=namespaces)
            
            md.title = find_text(record, 'gmd:identificationInfo/gmd:MD_DataIdentification/gmd:citation/gmd:CI_Citation/gmd:title/gco:CharacterString', namespaces=namespaces) or \
                    find_text(record, 'gmd:identificationInfo/srv:SV_ServiceIdentification/gmd:citation/gmd:CI_Citation/gmd:title/gco:CharacterString', namespaces=namespaces)

            md.abstract = find_text(record, 'gmd:identificationInfo/gmd:MD_DataIdentification/gmd:abstract/gco:CharacterString', namespaces=namespaces) or \
                find_text(record, 'gmd:identificationInfo/srv:SV_ServiceIdentification/gmd:abstract/gco:CharacterString', namespaces=namespaces)
            
            md.hierarchy_level = find_text(record, 'gmd:hierarchyLevel/gmd:MD_ScopeCode', namespaces=namespaces)

            mds += [md]
            _recordId += 1
        
        return mds


class DataSourceBuilder:

    def __init__(self, src_dir: str):
        self.src_dir = src_dir
        self.metadatas = []

    def load_data(self, xml_parser: callable = XmlParser.ceda):
        self.metadatas.clear()
        fps = glob.glob(os.path.join(self.src_dir, '*.xml'))
        
        for fp in fps:
            mds = xml_parser(fp)
            self.metadatas += mds
    
    def build_sqlite(self, db_fp: str = ":memory:") -> int:
        df = self.build_df()
        return df.to_sql(db_fp)
    
    def build_piclke(self, dst_fp: str = "metadata.pkl") -> None:
        df = self.build_df()
        df.to_pickle(dst_fp)

    def build_df(self) -> pd.DataFrame:
        data = [md.to_dict() for md in self.metadatas]
        return pd.DataFrame(data)
    
    def build_csv(self, dst_fp:str = "metadata.csv", index:bool = False) -> None:
        df = self.build_df()
        df.to_csv(dst_fp, index=index)

class Searcher:
    
    pretrained_models = [
        'msmarco-distilbert-base-dot-prod-v3', # this works okay
        'distilbert-base-nli-stsb-mean-tokens', # for short desciptions
        'all-mpnet-base-v2',
        'multi-qa-MiniLM-L6-cos-v1', # specifically trained for semantic search
        'msmarco-bert-base-dot-v5' # could try this one too
    ]
    
    def __init__(self, df: pd.DataFrame, model:str='multi-qa-MiniLM-L6-cos-v1'):
        if df is None:
            raise ValueError('df cannot be None')

        if model not in self.__class__.pretrained_models:
            raise ValueError(f'"{model}" is not a valid model. Please use one of the following: {self.pretrained_models}')
        
        self.df = df
        self.model_name = model
        self.model = SentenceTransformer(model)
        
    def embed_col(self, embed_col:str="abstract", **kwargs) -> None:
        '''embedding the text in the column, pickle the dataframe'''
        embeddings = self.model.encode(self.df.abstract.tolist(), show_progress_bar=True, **kwargs)
        self.df[f'{embed_col}_emb'] = embeddings.tolist()
        plk_fn = self.__get_emb_df_fp(col=embed_col)
        self.df.to_pickle(plk_fn)
    
    def __get_emb_df_fp(self, col:str="abstract"):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        emb_df_dir = os.path.join(current_dir, 'embedded_dataframes')
        if not os.path.exists(emb_df_dir):
            os.mkdir(emb_df_dir)
        
        return os.path.join(
            current_dir, 
            'embedded_dataframes',
            f'{self.model_name}_{col}.pkl')
    
    def get_embedded_df(self, embed_col:str="abstract"):
        plk_fp = self.__get_emb_df_fp(col=embed_col)
        return pd.read_pickle(plk_fp)
    
    def train_model(self):
        raise NotImplementedError()

    def search(self, query:str, col:str="abstract", k:int=5) -> pd.DataFrame:
        '''search for the query'''
        if f'{col}_emb' not in self.df.columns:
            plk_fp = self.__get_emb_df_fp(col=col)
            if os.path.exists(plk_fp):
                self.df = self.get_embedded_df(embed_col=col)
            else:
                self.embed_col(col)
        query_embedding = self.model.encode(query)
        cos_scores = util.cos_sim(query_embedding, self.df[f'{col}_emb'])
        top_results = torch.topk(cos_scores, k=k)
        
        row_idxs = top_results[1].tolist()[0]
        scores = top_results[0].tolist()[0]
        
        results = self.df.iloc[row_idxs].copy()
        results["scores"] = scores
        
        return results


if __name__ == '__main__':
    src_dir = r'C:\Users\ChrisLam\Downloads\NERC-2023-02-18\2023-02-18\2.0.2\GMD'
    dst_fp=r"C:\Users\ChrisLam\Documents\PythonNLP\app\metadata_dataframes\2023-02-18-GMD-metadata.pkl"
    builder = DataSourceBuilder(src_dir=src_dir)
    builder.load_data(xml_parser=XmlParser.gmd)
    df = builder.build_df()
    print(df.head())
    
    builder.build_piclke(dst_fp=dst_fp)
    
    TextCleaner.clean_column(df, col='title', clean_func=TextCleaner.default_clean)
    TextCleaner.clean_column(df, col='abstract', clean_func=TextCleaner.default_clean)
    
    searcher = Searcher(df = df, model = 'multi-qa-MiniLM-L6-cos-v1')
    searcher.embed_col(embed_col="title")
    searcher.embed_col(embed_col="abstract")
    
    # testing
    
    r1 = searcher.search(
        query = 'find me datasets showing precipitation in the uk for the last 20 years',
        col = "abstract",
        k = 5
    )
    print(r1)
    
    r2 = searcher.search(
        query = 'find me datasets showing precipitation in the uk for the last 20 years',
        col = "title",
        k = 5
    )
    print(r2)
    



    
