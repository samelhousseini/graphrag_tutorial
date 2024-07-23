import sys
import os
from datetime import datetime
import json

from itertools import groupby
import time
import graphrag
import subprocess
from env_vars import *
from dotenv import load_dotenv, set_key
import yaml
import logging
import time
from typing import Any
import pandas as pd
from utils.file_utils import *

import tiktoken

from graphrag.query.context_builder.builders import LocalContextBuilder
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
from graphrag.query.llm.base import BaseLLM, BaseLLMCallback
from graphrag.query.llm.text_utils import num_tokens
from graphrag.query.structured_search.base import BaseSearch, SearchResult
from graphrag.query.structured_search.local_search.system_prompt import (LOCAL_SEARCH_SYSTEM_PROMPT,)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import (store_entity_semantic_embeddings,)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (LocalSearchMixedContext,)
from graphrag.query.structured_search.global_search.community_context import (GlobalCommunityContext,)
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore


global_context_builder_params = {
    "use_community_summary": False,  # False means using full community reports. True means using community short summaries.
    "shuffle_data": True,
    "include_community_rank": True,
    "min_community_rank": 0,
    "community_rank_name": "rank",
    "include_community_weight": True,
    "community_weight_name": "occurrence weight",
    "normalize_community_weight": True,
    "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
    "context_name": "Reports",
}

global_map_llm_params = {
    "max_tokens": 1000,
    "temperature": 0.0,
    "response_format": {"type": "json_object"},
}

global_reduce_llm_params = {
    "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
    "temperature": 0.0,
}

local_context_params = {
    "text_unit_prop": 0.5,
    "community_prop": 0.1,
    "conversation_history_max_turns": 5,
    "conversation_history_user_turns_only": True,
    "top_k_mapped_entities": 10,
    "top_k_relationships": 10,
    "include_entity_rank": True,
    "include_relationship_weight": True,
    "include_community_rank": False,
    "return_candidate_context": False,
    "embedding_vectorstore_key": EntityVectorStoreKey.ID,  # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
    "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
}

local_llm_params = {
    "max_tokens": 2_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000=1500)
    "temperature": 0.0,
}



## Please check the Leiden Algorithm for community detection
# https://en.wikipedia.org/wiki/Leiden_algorithm



class GraphRagHelper():

    def __init__(self, project_dir) -> None:
        self.project_dir = project_dir
        self.output_dir = os.path.join(project_dir, "output")
        self.input_dir = os.path.join(project_dir, "input")
        self.prompt_dir = os.path.join(project_dir, "prompts")

        self.env_path = os.path.join(self.project_dir, '.env')
        self.yaml_path = os.path.join(self.project_dir, 'settings.yaml')

        self.local_dict = {}
        self.global_dict = {}

        self.token_encoder = tiktoken.get_encoding("cl100k_base")

        self.llm = ChatOpenAI(
            api_key=AZURE_OPENAI_KEY,
            model=AZURE_OPENAI_MODEL,
            deployment_name=AZURE_OPENAI_MODEL,
            api_type=OpenaiApiType.AzureOpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
            api_version=AZURE_OPENAI_API_VERSION,
            api_base=f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com",
            max_retries=20,
        )

        self.embedder = OpenAIEmbedding(
            api_key=AZURE_OPENAI_EMBEDDING_MODEL_RESOURCE_KEY,
            model=AZURE_OPENAI_EMBEDDING_MODEL,
            deployment_name=AZURE_OPENAI_EMBEDDING_MODEL,
            api_type=OpenaiApiType.AzureOpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
            api_version=AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION,
            api_base=f"https://{AZURE_OPENAI_EMBEDDING_MODEL_RESOURCE}.openai.azure.com",
            max_retries=20,
        )



    def initialize(self):
        # Define the command to run
        command = ['python', '-m', 'graphrag.index', '--init', '--root', self.project_dir]

        # Run the command using subprocess and capture the output
        print(f"Subprocessing command: {command}")
        result = subprocess.run(command, capture_output=True, text=True)

        # Print the standard output and standard error
        print("Standard Output:")
        print(result.stdout)

        print("Standard Error:")
        print(result.stderr)

        load_dotenv(self.env_path)
        set_key(self.env_path, 'GRAPHRAG_API_KEY', AZURE_OPENAI_KEY)

        settings = read_yaml(self.yaml_path)

        try:
            settings['llm']['type'] = 'azure_openai_chat'
            settings['llm']['deployment_name'] = AZURE_OPENAI_MODEL
            settings['llm']['model'] = AZURE_OPENAI_MODEL
            settings['llm']['api_base'] = f"https://{os.getenv('AZURE_OPENAI_RESOURCE')}.openai.azure.com"
            settings['llm']['api_version'] = AZURE_OPENAI_API_VERSION
            
            settings['embeddings']['llm']['api_key'] = AZURE_OPENAI_EMBEDDING_MODEL_RESOURCE_KEY
            settings['embeddings']['llm']['type'] = 'azure_openai_embedding'
            settings['embeddings']['llm']['deployment_name'] = AZURE_OPENAI_EMBEDDING_MODEL
            settings['embeddings']['llm']['model'] = AZURE_OPENAI_EMBEDDING_MODEL
            settings['embeddings']['llm']['api_base'] = f"https://{os.getenv('AZURE_OPENAI_EMBEDDING_MODEL_RESOURCE')}.openai.azure.com"
            settings['embeddings']['llm']['api_version'] = AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION

            write_yaml(self.yaml_path, settings)

        except Exception as e:
            print(e)

        ret = {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'status': (result.stderr == '') and ('Initializing project' in result.stdout)
        }

        return ret
    

    def load_env(self):
        load_dotenv(self.env_path)

        # Get the current environment variables
        env = os.environ.copy()

        # Add the custom environment variables from the .env file
        env.update({
            'GRAPHRAG_API_KEY': AZURE_OPENAI_KEY
        })
    
        return env
    
    def prompt_fine_tune(self):
        # Define the command to run
        command = ['python', '-m', 'graphrag.prompt_tune', '--root', self.project_dir, '--no-entity-types']

        # Run the command using subprocess and capture the output
        print(f"Subprocessing command: {command}")
        result = subprocess.run(command, env=self.load_env(), capture_output=True, text=True)

        # Print the standard output and standard error
        print("Standard Output:")
        print(result.stdout)

        print("Standard Error:")
        print(result.stderr)

        ret = {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'status': (result.stderr == '') and ('stored in folder' in result.stdout)
        }

        return ret
    

    # Function to convert folder name to datetime object
    def folder_name_to_datetime(self, folder_name):
        try:
            return datetime.strptime(folder_name, '%Y%m%d-%H%M%S')
        except ValueError:
            return None
        
    
    def get_latest_run_folder(self):
        # List all items in the "output" folder and filter out only directories
        all_folders = [f for f in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, f))]

        # Create a list of tuples with folder names and their corresponding datetime objects
        folders_with_dates = [(folder, self.folder_name_to_datetime(folder)) for folder in all_folders]

        # Filter out any folders that didn't match the timestamp format
        valid_folders_with_dates = [item for item in folders_with_dates if item[1] is not None]

        most_recent_folder = None
        # Find the most recent folder
        if valid_folders_with_dates:
            most_recent_folder = max(valid_folders_with_dates, key=lambda x: x[1])[0]
            print(f"The most recent folder is: {most_recent_folder}")
        else:
            print("No valid timestamp folders found in the 'output' directory.")

        return most_recent_folder


    def index_data(self):
        # delete_folder(self.output_dir)
        most_recent_folder = self.get_latest_run_folder()
        most_recent_path = os.path.join(self.output_dir, most_recent_folder, "artifacts")
        delete_files_with_extension(most_recent_path, '.parquet')

        os.makedirs(self.output_dir, exist_ok=True)

        if most_recent_folder is None:
            # Define the command to run
            command = ['python', '-m', 'graphrag.index', '--root', self.project_dir]
        else:
            command = ['python', '-m', 'graphrag.index', '--root', self.project_dir, '--resume', most_recent_folder]

        # Run the command using subprocess and capture the output
        print(f"Subprocessing command: {command}")
        result = subprocess.run(command, env=self.load_env(), capture_output=True, text=True)

        # Print the standard output and standard error
        print("Standard Output:")
        print(result.stdout)

        print("Standard Error:")
        print(result.stderr)

        ret = {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'status': (result.stderr == '') and ('All workflows completed successfully' in result.stdout)
        }

        return ret
    

    def load_data(self, source_data_dir, copy_from_accelerator = False):

        if copy_from_accelerator:
            all_folders = [f for f in os.listdir(source_data_dir) if (os.path.isdir(os.path.join(source_data_dir, f))) and ('.' in f)]
            all_data_files = [os.path.join(source_data_dir, f, replace_extension(os.path.basename(f), '.txt')) for f in all_folders]
            print(all_folders)
            print("\n", all_data_files)
            for f in all_data_files: copy_file(f, self.input_dir)
        else:
            copy_files(source_data_dir, self.input_dir)


    def load_local_search_engine(self, community_level = 2):
        
        if community_level not in self.local_dict.keys():
            LANCEDB_URI = f"{self.project_dir}/lancedb"
            COMMUNITY_REPORT_TABLE = "create_final_community_reports"
            ENTITY_TABLE = "create_final_nodes"
            ENTITY_EMBEDDING_TABLE = "create_final_entities"
            RELATIONSHIP_TABLE = "create_final_relationships"
            COVARIATE_TABLE = "create_final_covariates"
            TEXT_UNIT_TABLE = "create_final_text_units"

            most_recent_folder = self.get_latest_run_folder()
            most_recent_path = os.path.join(self.output_dir, most_recent_folder, "artifacts")

            print("Most recent path: ", most_recent_path)
            print("Entity Table path: ", f"{most_recent_path}/{ENTITY_TABLE}.parquet")

            entity_df = pd.read_parquet(f"{most_recent_path}/{ENTITY_TABLE}.parquet")
            entity_embedding_df = pd.read_parquet(f"{most_recent_path}/{ENTITY_EMBEDDING_TABLE}.parquet")
            entities = read_indexer_entities(entity_df, entity_embedding_df, community_level)
            description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings",)
            description_embedding_store.connect(db_uri=LANCEDB_URI)
            entity_description_embeddings = store_entity_semantic_embeddings(entities=entities, vectorstore=description_embedding_store)
            relationship_df = pd.read_parquet(f"{most_recent_path}/{RELATIONSHIP_TABLE}.parquet")
            relationships = read_indexer_relationships(relationship_df)
            report_df = pd.read_parquet(f"{most_recent_path}/{COMMUNITY_REPORT_TABLE}.parquet")
            reports = read_indexer_reports(report_df, entity_df, community_level)
            text_unit_df = pd.read_parquet(f"{most_recent_path}/{TEXT_UNIT_TABLE}.parquet")
            text_units = read_indexer_text_units(text_unit_df)

            covariate_df = (
                pd.read_parquet(f"{most_recent_path}/{COVARIATE_TABLE}.parquet")
                if os.path.exists(f"{most_recent_path}/{COVARIATE_TABLE}.parquet")
                else None
            )
            covariates = (
                read_indexer_covariates(covariate_df)
                if covariate_df is not None
                else []
            )

            covariates = {"claims": covariates}

            self.local_dict = {}
            self.local_dict[community_level] = {
                'entities': entities,
                'description_embedding_store': description_embedding_store,
                'entity_description_embeddings': entity_description_embeddings,
                'relationships': relationships,
                'covariates': covariates,
                'reports': reports,
                'text_units': text_units
            }

            self.local_context_builder = LocalSearchMixedContext(
                community_reports=reports,
                text_units=text_units,
                entities=entities,
                relationships=relationships,
                covariates=covariates,
                entity_text_embeddings=description_embedding_store,
                embedding_vectorstore_key=EntityVectorStoreKey.ID,  # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
                text_embedder=self.embedder,
                token_encoder=self.token_encoder,
            )

            self.local_search_engine = LocalSearch(
                llm=self.llm,
                context_builder=self.local_context_builder,
                token_encoder=self.token_encoder,
                llm_params=local_llm_params,
                context_builder_params=local_context_params,
                response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
            )


    def load_global_search_engine(self, community_level = 2):
        if community_level not in self.global_dict.keys():
            most_recent_folder = self.get_latest_run_folder()
            most_recent_path = os.path.join(self.output_dir, most_recent_folder, "artifacts")

            print("Most recent path: ", most_recent_path)

            # parquet files generated from indexing pipeline
            COMMUNITY_REPORT_TABLE = "create_final_community_reports"
            ENTITY_TABLE = "create_final_nodes"
            ENTITY_EMBEDDING_TABLE = "create_final_entities"

            entity_df = pd.read_parquet(f"{most_recent_path}/{ENTITY_TABLE}.parquet")
            report_df = pd.read_parquet(f"{most_recent_path}/{COMMUNITY_REPORT_TABLE}.parquet")
            entity_embedding_df = pd.read_parquet(f"{most_recent_path}/{ENTITY_EMBEDDING_TABLE}.parquet")

            reports = read_indexer_reports(report_df, entity_df, community_level)
            entities = read_indexer_entities(entity_df, entity_embedding_df, community_level)

            self.global_dict = {}
            self.global_dict[community_level] = {
                'entities': entities,
                'reports': reports
            }

            self.global_context_builder = GlobalCommunityContext(
                community_reports=reports,
                entities=entities,  # default to None if you don't want to use community weights for ranking
                token_encoder=self.token_encoder,
            )

            self.global_search_engine = GlobalSearch(
                llm=self.llm,
                context_builder=self.global_context_builder,
                token_encoder=self.token_encoder,
                max_data_tokens=12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
                map_llm_params=global_map_llm_params,
                reduce_llm_params=global_reduce_llm_params,
                allow_general_knowledge=False,  # set this to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
                json_mode=True,  # set this to False if your LLM model does not support JSON mode.
                context_builder_params=global_context_builder_params,
                concurrent_coroutines=32,
                response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
            )
                    


    def local_search(self, query, community_level = 2, context_only=False):
        self.load_local_search_engine(community_level=community_level)

        if context_only:
            context_text, context_records = self.local_context_builder.build_context(
                query=query,
                conversation_history=None,
                **local_context_params,
            )
            return {'context_text': context_text, 'context_records': context_records}
        else:
            result = self.local_search_engine.search(query)
            print(result.response)
            return {'response': result.response}


    async def global_search(self, query, community_level = 2, context_only=False):
        self.load_global_search_engine(community_level=community_level)

        if context_only:
            context_text, context_records = self.global_context_builder.build_context(
                query=query,
                conversation_history=None,
                **global_context_builder_params,
            )
            return {'context_text': context_text, 'context_records': context_records}
        else:
            result = await self.global_search_engine.asearch(query)
            print(result.response)
            return {'response': result.response}
